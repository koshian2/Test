import matplotlib.pyplot as plt
import os
import numpy as np
from libs.vgg16 import deprocess_image

def add_mask(ground_truth_batch, blurred_mask_batch):
    assert ground_truth_batch.ndim == 4 and blurred_mask_batch.ndim == 4
    assert ground_truth_batch.dtype == np.uint8 and blurred_mask_batch.dtype == np.uint8
    x = ground_truth_batch.astype(np.int16) + blurred_mask_batch.astype(np.int16)
    return np.clip(x, 0, 255).astype(np.uint8)

# 画像をタイルして保存
def tile_images(masked_image, unmasked_image, grand_truth, filepath, title):
    assert masked_image.shape[0] >= 3 and unmasked_image.shape[0] >= 3 and grand_truth.shape[0] >= 3
    assert masked_image.dtype == np.float32 and grand_truth.dtype == np.float32 and unmasked_image.dtype == np.float32
    # 色空間をもとに戻す
    y_masked = deprocess_image(masked_image)
    y_unmasked = deprocess_image(unmasked_image)
    y_gt = deprocess_image(grand_truth)

    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.mkdir(directory)

    def plot_subplot(index, image):
        ax = plt.subplot(3, 3, index)
        ax.imshow(image)
        ax.axis("off")

    plt.subplots_adjust(hspace=0.02, wspace=0.02, top=0.95, bottom=0.02, left=0.02, right=0.98)
    plt.clf()
    for i in range(3):
        plot_subplot(3*i+1, y_masked[i])
        plot_subplot(3*i+2, y_unmasked[i])
        plot_subplot(3*i+3, y_gt[i])
    plt.suptitle(title)
    plt.savefig(filepath)
