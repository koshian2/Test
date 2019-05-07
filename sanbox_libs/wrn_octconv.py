from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from oct_conv2d import OctConv2D
import numpy as np

def residual_block_oct(high_x, low_x, in_filter, out_filter, alpha, stride, activate_before_residual=False):
    if activate_before_residual:
        high_x = layers.BatchNormalization()(high_x)
        high_x = layers.Activation("relu")(high_x)
        low_x = layers.BatchNormalization()(low_x)
        low_x = layers.Activation("relu")(low_x)
        orig_high, orig_low = high_x, low_x
    else:
        orig_high, orig_low = high_x, low_x

    block_high, block_low = high_x, low_x
    if not activate_before_residual:
        block_high = layers.BatchNormalization()(block_high)
        block_high = layers.Activation("relu")(block_high)
        block_low = layers.BatchNormalization()(block_low)
        block_low = layers.Activation("relu")(block_low)

    # Since centers are shifted when stride > 1(see OctConv paper), add average pooling for downsampling
    if stride > 1:
        block_high = layers.AveragePooling2D(stride)(block_high)
        block_low = layers.AveragePooling2D(stride)(block_low)
    block_high, block_low = OctConv2D(out_filter, alpha)([block_high, block_low])

    block_high = layers.BatchNormalization()(block_high)
    block_high = layers.Activation("relu")(block_high)
    block_low = layers.BatchNormalization()(block_low)
    block_low = layers.Activation("relu")(block_low)

    block_high, block_low = OctConv2D(out_filter, alpha)([block_high, block_low])

    if in_filter != out_filter:
        orig_high = layers.AveragePooling2D(stride)(orig_high)
        orig_high = layers.Lambda(zero_pad, 
                    arguments={"in_filter":int(in_filter*(1-alpha)), 
                               "out_filter":int(out_filter*(1-alpha))})(orig_high)
        orig_low = layers.AveragePooling2D(stride)(orig_low)
        orig_low = layers.Lambda(zero_pad, 
                    arguments={"in_filter":int(in_filter*alpha), 
                               "out_filter":int(out_filter*alpha)})(orig_low)
    block_high = layers.Add()([orig_high, block_high])
    block_low = layers.Add()([orig_low, block_low])
    return block_high, block_low

def zero_pad(inputs, in_filter=1, out_filter=1):
  """Zero pads `input` tensor to have `out_filter` number of filters."""
  outputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                            [(out_filter - in_filter) // 2,
                             (out_filter - in_filter) // 2]])
  return outputs

def _res_add_oct(in_filter, out_filter, alpha, stride, high_x, low_x, orig_high, orig_low):
    if in_filter != out_filter:
        orig_high = layers.AveragePooling2D(stride)(orig_high)
        orig_high = layers.Lambda(zero_pad, 
                    arguments={"in_filter":int(in_filter*(1-alpha)), 
                               "out_filter":int(out_filter*(1-alpha))})(orig_high)
        orig_low = layers.AveragePooling2D(stride)(orig_low)
        orig_low = layers.Lambda(zero_pad, 
                    arguments={"in_filter":int(in_filter*alpha), 
                               "out_filter":int(out_filter*alpha)})(orig_low)
    block_high = layers.Add()([orig_high, high_x])
    block_low = layers.Add()([orig_low, low_x])
    return block_high, orig_high, block_low, orig_low

def build_wrn_oct_model(alpha, num_classes=10, wrn_size=160):
    kernel_size = wrn_size
    filter_size = 3
    num_blocks_per_resnet = 4
    filters = [
        min(kernel_size, 16), kernel_size, kernel_size*2, kernel_size*4
    ]
    strides = [1,2,2]

    # first conv
    input = layers.Input((32,32,3))
    high = input
    low = layers.AveragePooling2D(2)(input)
    high, low = OctConv2D(filters[0], alpha)([high, low])
    first_high, first_low = high, low # Res from the begging
    orig_high, orig_low = high, low # Res from previous block

    for block_num in range(1, 4):
        activate_before_residual = True if block_num == 1 else False
        high, low = residual_block_oct(high, low, 
                        filters[block_num-1], filters[block_num], alpha, 
                        strides[block_num-1], activate_before_residual=activate_before_residual)
        for i in range(1, num_blocks_per_resnet):
            high, low = residual_block_oct(high, low,
                            filters[block_num], filters[block_num], alpha, 
                            1, activate_before_residual=False)
        high, orig_high, low, orig_low = _res_add_oct(filters[block_num-1], filters[block_num],
                                            alpha, strides[block_num-1], high, low, orig_high, orig_low)

    final_stride_val = int(np.prod(strides))
    high, _, low, _ = _res_add_oct(filters[0], filters[3], alpha, final_stride_val, 
                            high, low, first_high, first_low)
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)

    high = layers.AveragePooling2D(2)(high)
    x = layers.Concatenate()([high, low])
    x = layers.Conv2D(filters[3], 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    return Model(input, x)
