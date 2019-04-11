import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.contrib.tpu.python.tpu import keras_support

from libs.pconv_layer import PConv2D
from libs.loss_layer import LossLayer
from libs.vgg16 import extract_vgg_features
from libs.utils import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# 画像のカラースケールはNan対策でtf[-1,1]とする。VGGは別に処理する。
def conv_bn_relu(image_in, mask_in, filters, kernel_size, 
                 downsampling=1, upsampling=1, act="relu",
                 concat_img=None, concat_mask=None, reps=1):
    assert not (concat_img is None)^(concat_mask is None) # XORは常にFalse
    # Upsamplingする場合
    if upsampling > 1:
        conv = layers.Lambda(upsampling2d_tpu, arguments={"scale":upsampling})(image_in)
        mask = layers.Lambda(upsampling2d_tpu, arguments={"scale":upsampling})(mask_in)
    else:
        conv, mask = image_in, mask_in
    if concat_img is not None and concat_mask is not None:
        conv = layers.Concatenate()([conv, concat_img])
        mask = layers.Concatenate()([mask, concat_mask])
        # 計算量削減のために1x1 Convを入れる
        conv, mask = PConv2D(filters=filters, kernel_size=1)([conv, mask])
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation("relu")(conv)

    for i in range(reps):
        stride = downsampling if i == 0 else 1
        # strideでダウンサンプリング
        conv, mask = PConv2D(filters=filters, kernel_size=kernel_size, 
                             padding="same", strides=stride)([conv, mask])
        # Image側だけBN->ReLUを入れる
        conv = layers.BatchNormalization()(conv)
        if act == "relu":
            conv = layers.Activation("relu")(conv)
        elif act == "prelu":
            conv = layers.PReLU()(conv)
        elif act == "custom_tanh":
            # 元の画像の白を黒に変えるには、tanhの2倍のスケール[-2,2]が必要
            conv = layers.Lambda(lambda x: 2*K.tanh(x), name="unmasked")(conv)
    return conv, mask

def convert_caffe_color_space(tf_color_input):
    # 画像の出力層の活性化関数（前処理に合わせる）
    # [-1,1] -> [0,255]のスケール
    x = tf_color_input * 127.5 + 127.5
    # RGB -> BGR
    x = x[:,:,:,::-1]
    # カラーチャンネルごとのシフト
    mean = np.array([103.939, 116.779, 123.68]).reshape(1,1,1,-1)
    x = x - K.variable(mean)
    return x

def upsampling2d_tpu(inputs, scale=2):
    x = K.repeat_elements(inputs, scale, axis=1)
    x = K.repeat_elements(x, scale, axis=2)
    return x

def create_train_pconv_unet():
    input_image = layers.Input((320,240,3))
    input_mask = layers.Input((320,240,3))
    input_groundtruth = layers.Input((320,240,3))

    ## U-Net
    # Encoder
    conv1, mask1 = conv_bn_relu(input_image, input_mask, 
                                filters=32, kernel_size=3, downsampling=1, reps=2) # 320x240
    conv2, mask2 = conv_bn_relu(conv1, mask1,
                                filters=64, kernel_size=7, downsampling=5, reps=1)
    conv2, mask2 = conv_bn_relu(conv2, mask2,
                                filters=64, kernel_size=3, downsampling=1, reps=1) # 64x48
    conv3, mask3 = conv_bn_relu(conv2, mask2,
                                filters=128, kernel_size=3, downsampling=2, reps=2) # 32x24
    conv4, mask4 = conv_bn_relu(conv3, mask3,
                                filters=256, kernel_size=3, downsampling=2, reps=2) # 16x12
    conv5, mask5 = conv_bn_relu(conv4, mask4,
                                filters=512, kernel_size=3, downsampling=2, reps=2) # 8x6
    ## Decoder
    img, mask = conv_bn_relu(conv5, mask5,
                             filters=256, kernel_size=3, upsampling=2, reps=2,
                             concat_img=conv4, concat_mask=mask4) # 16x12
    img, mask = conv_bn_relu(img, mask,
                             filters=128, kernel_size=3, upsampling=2, reps=2,
                             concat_img=conv3, concat_mask=mask3) # 32x24
    img, mask = conv_bn_relu(img, mask,
                             filters=64, kernel_size=3, upsampling=2, reps=2,
                             concat_img=conv2, concat_mask=mask2) # 64x48
    img, mask = conv_bn_relu(img, mask,
                             filters=32, kernel_size=7, upsampling=5, reps=1,
                             concat_img=conv1, concat_mask=mask1) # 320x240
    img, mask = conv_bn_relu(img, mask,
                             filters=32, kernel_size=3, upsampling=1, reps=2)
    img, mask = conv_bn_relu(img, mask,
                             filters=3, kernel_size=1, reps=1, act="custom_tanh") # 差分出力
    # skip connection
    img = layers.Add()([img, input_image]) # 収束が早くなるはず
    img = layers.Lambda(lambda x: K.clip(x, -1.0, 1.0))(img)

    ## 損失関数
    # マスクしていない部分の真の画像＋マスク部分の予測画像
    y_comp = layers.Lambda(lambda inputs: inputs[0]*inputs[1] + (1-inputs[0])*inputs[2])(
        [input_mask, input_groundtruth, img])
    # Caffeカラースケールに変換
    vgg_in_pred = layers.Lambda(convert_caffe_color_space)(img)
    vgg_in_groundtruth = layers.Lambda(convert_caffe_color_space)(input_groundtruth)
    vgg_in_comp = layers.Lambda(convert_caffe_color_space)(y_comp)
    # vggの特徴量
    vgg_pred_1, vgg_pred_2, vgg_pred_3 = extract_vgg_features(vgg_in_pred, (320,240,3), 0)
    vgg_true_1, vgg_true_2, vgg_true_3 = extract_vgg_features(vgg_in_groundtruth, (320,240,3), 1)
    vgg_comp_1, vgg_comp_2, vgg_comp_3 = extract_vgg_features(vgg_in_comp, (320,240,3), 2)
    # 画像＋損失
    join = LossLayer()([input_mask,
                        img, input_groundtruth, y_comp,
                        vgg_pred_1, vgg_pred_2, vgg_pred_3,
                        vgg_true_1, vgg_true_2, vgg_true_3,
                        vgg_comp_1, vgg_comp_2, vgg_comp_3])
    # lossやmetricsの表示がうまくいかないので出力は1つにする
    model = Model([input_image, input_mask, input_groundtruth], join) # このモデルは>100MBだが、推論用モデルは93MB

    return model

# 損失関数側だけ取る
def identity_loss(y_true, y_pred):
    return K.mean(y_pred[:,:,:,3], axis=(1,2))

def PSNR(y_true, y_pred):
    # 参考：https://ja.wikipedia.org/wiki/%E3%83%94%E3%83%BC%E3%82%AF%E4%BF%A1%E5%8F%B7%E5%AF%BE%E9%9B%91%E9%9F%B3%E6%AF%94
    pic_gt = y_true[:,:,:,:3]
    pic_pred = y_pred[:,:,:,:3]
    return 20 * K.log(2.0) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(pic_gt-pic_pred), axis=(1,2,3))) / K.log(10.0) 

def data_generator(total_data, filter_indices, batch_size, shuffle):
    image_cache, blurred_cache, mask_cache = [], [], []
    while True:
        indices = filter_indices.copy()
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            image_cache.append(total_data[i,:,:,:3])
            blurred_cache.append(total_data[i,:,:,3:4])
            mask_cache.append(total_data[i,:,:,4:])
            if len(image_cache) == batch_size:
                batch_gt = np.asarray(image_cache, np.uint8)
                batch_blurred = np.asarray(blurred_cache, np.uint8) * np.ones((1,1,1,3), np.uint8)
                batch_mask = np.asarray(mask_cache, np.uint8) * np.ones((1,1,1,3), np.float32) #マスクも3ch
                image_cache, blurred_cache, mask_cache = [], [], []
                # マスク済み画像を作る
                batch_masked_image = add_mask(batch_gt, batch_blurred)
                # 前処理
                batch_gt = preprocess_image(batch_gt)
                batch_masked_image = preprocess_image(batch_masked_image)
                batch_mask = 1.0 - batch_mask / 255.0 # マスク部分は0、画像部分は1（ハマるので注意）
                # yはgt+dummy
                batch_y = np.zeros((batch_gt.shape[0], batch_gt.shape[1], batch_gt.shape[2], 4), np.float32)
                batch_y[:,:,:,:3] = batch_gt

                yield [batch_masked_image, batch_mask, batch_gt], batch_y
                
def load_data():
    data = np.load("oppai_dataset/oppai.npz")["data"]
    train, test = train_test_split(np.arange(len(data)), test_size=0.15, random_state=114514)
    # メモリ対策で全体データとインデックスとする
    return data, train, test

class SamplingCallback(Callback):
    def __init__(self, model, data, test_indices):
        self.model = model
        self.data = data
        self.test_indices = test_indices
        self.min_val_loss = np.inf

    def on_epoch_end(self, epoch, logs):
        # エポックごとにマスク修復の訓練の進みを可視化して保存
        gen = data_generator(self.data, self.test_indices, 8, False)
        for i in range(10):
            [masked, mask, gt], _ = next(gen)
            unmasked = self.model.predict([masked, mask, gt])[:,:,:,:3]
            tile_images(masked, unmasked, gt, f"sampling/epoch_{epoch:03}_{i}.png",
                        f"Masked / Pred / Ground Truth (epoch={epoch:03} {i+1}/10)")
        # モデルの保存
        if self.min_val_loss > logs["val_loss"]:
            print(f"Val loss improved {self.min_val_loss:.04} to {logs['val_loss']:.04}")
            self.min_val_loss = logs["val_loss"]
            self.model.save_weights("oppai_train.hdf5" ,save_format="h5")

def train():
    data, train_ind, test_ind = load_data()
    print(train_ind.shape, test_ind.shape)

    model = create_train_pconv_unet()
    model.summary()
    model.compile(tf.train.RMSPropOptimizer(5e-5), identity_loss, [PSNR])

    # TPUモデルに変換
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    batch_size=8
    cb = SamplingCallback(model, data, test_ind)

    model.fit_generator(data_generator(data, train_ind, batch_size, True),
                        steps_per_epoch=len(train_ind)//batch_size,
                        validation_data=data_generator(data, test_ind, batch_size, False),
                        validation_steps=len(test_ind)//batch_size,
                        callbacks=[cb], epochs=70, max_queue_size=1)

if __name__ == "__main__":
    K.clear_session()
    train()
