from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

# https://github.com/tensorflow/models/blob/master/research/autoaugment/wrn.py
# Port from TensorFlow to Keras

def residual_block(x, in_filter, out_filter, stride, activate_before_residual=False):
    if activate_before_residual:
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        orig_x = x
    else:
        orig_x = x

    block_x = x
    if not activate_before_residual:
        block_x = layers.BatchNormalization()(block_x)
        block_x = layers.Activation("relu")(block_x)

    block_x = layers.Conv2D(out_filter, 3, padding="same", strides=stride)(block_x)

    block_x = layers.BatchNormalization()(block_x)
    block_x = layers.Activation("relu")(block_x)
    block_x = layers.Conv2D(out_filter, 3, padding="same", strides=1)(block_x)

    if in_filter != out_filter:
        orig_x = layers.AveragePooling2D(stride)(orig_x)
        orig_x = layers.Lambda(zero_pad, arguments={"in_filter":in_filter, "out_filter":out_filter})(orig_x)
    x = layers.Add()([orig_x, block_x])
    return x

def zero_pad(inputs, in_filter=1, out_filter=1):
  """Zero pads `input` tensor to have `out_filter` number of filters."""
  outputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                            [(out_filter - in_filter) // 2,
                             (out_filter - in_filter) // 2]])
  return outputs

def _res_add(in_filter, out_filter, stride, x, orig_x):
    if in_filter != out_filter:
        orig_x = layers.AveragePooling2D(stride)(orig_x)
        orig_x = layers.Lambda(zero_pad, arguments={"in_filter":in_filter, "out_filter":out_filter})(orig_x)
    x = layers.Add()([x, orig_x])
    return x, orig_x

def build_wrn_model(num_classes=10, wrn_size=160):
    kernel_size = wrn_size
    filter_size = 3
    num_blocks_per_resnet = 4
    filters = [
        min(kernel_size, 16), kernel_size, kernel_size*2, kernel_size*4
    ]
    strides = [1,2,2]

    # first conv
    input = layers.Input((32,32,3))
    x = layers.Conv2D(filters[0], filter_size, padding="same")(input)
    first_x = x # Res from the begging
    orig_x = x # Res from previous block

    for block_num in range(1, 4):
        activate_before_residual = True if block_num == 1 else False
        x = residual_block(x, filters[block_num-1], 
                filters[block_num], strides[block_num-1],
                activate_before_residual=activate_before_residual)
        for i in range(1, num_blocks_per_resnet):
            x = residual_block(x, filters[block_num], filters[block_num], 1,
                activate_before_residual=False)
        x, orig_x = _res_add(filters[block_num-1], filters[block_num],
                            strides[block_num-1], x, orig_x)

    final_stride_val = int(np.prod(strides))
    x, _ = _res_add(filters[0], filters[3], final_stride_val, x, first_x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    return Model(input, x)
