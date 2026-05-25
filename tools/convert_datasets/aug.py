import cv2
import numpy as np
import random
import os
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.transform import rotate, resize

# 1. 几何变换
def geometric_augmentation(image, label, rotation_range=20, scale_range=0.1, flip_prob=0.5):
    """
    对图像和标签进行几何变换（旋转、缩放、翻转）。
    """
    # 随机旋转
    angle = random.uniform(-rotation_range, rotation_range)
    image = rotate(image, angle, mode='constant', cval=0)
    label = rotate(label, angle, mode='constant', cval=0)

    # # 随机缩放
    # scale = random.uniform(1 - scale_range, 1 + scale_range)
    # h, w = image.shape[:2]
    # new_h, new_w = int(h * scale), int(w * scale)
    # image = resize(image, (new_h, new_w), mode='constant', cval=0)
    # label = resize(label, (new_h, new_w), mode='constant', cval=0)

    # 随机翻转
    if random.random() < flip_prob:
        image = np.fliplr(image)
        label = np.fliplr(label)

    return image, label

# 2. 灰度变换
def intensity_augmentation(image, brightness_range=0.1, contrast_range=0.1):
    """
    对图像进行灰度变换（亮度、对比度调整）。
    """
    # 随机亮度调整
    brightness = random.uniform(-brightness_range, brightness_range)
    image = np.clip(image + brightness, 0, 1)

    # 随机对比度调整
    contrast = random.uniform(1 - contrast_range, 1 + contrast_range)
    image = np.clip(image * contrast, 0, 1)

    return image

# 3. 弹性形变
def elastic_deformation(image, label, alpha=1000, sigma=30):
    """
    对图像和标签进行弹性形变。
    """
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    image = map_coordinates(image, indices, order=1).reshape(shape)
    label = map_coordinates(label, indices, order=1).reshape(shape)

    return image, label

# 4. CutMix增强
def cutmix_augmentation(image1, label1, image2, label2):
    """
    对两张图像和标签进行CutMix增强。
    """
    h, w = image1.shape[:2]
    lam = np.random.beta(1.0, 1.0)  # CutMix比例
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    bbx1 = np.clip(cx - int(w * np.sqrt(lam) / 2), 0, w)
    bby1 = np.clip(cy - int(h * np.sqrt(lam) / 2), 0, h)
    bbx2 = np.clip(cx + int(w * np.sqrt(lam) / 2), 0, w)
    bby2 = np.clip(cy + int(h * np.sqrt(lam) / 2), 0, h)

    # CutMix操作
    image_mixed = image1.copy()
    label_mixed = label1.copy()
    image_mixed[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
    label_mixed[bby1:bby2, bbx1:bbx2] = label2[bby1:bby2, bbx1:bbx2]

    return image_mixed, label_mixed

# 加载图像和标签对
def load_image_label_pairs(image_dir, label_dir):
    """
    从指定目录加载图像和标签对。
    假设图像和标签的文件名一一对应。
    """
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    image_label_pairs = []
    for img_file, lbl_file in zip(image_files, label_files):
        image_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, lbl_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        image_label_pairs.append((image, label))
    return image_label_pairs


# 保存增强后的图像和标签
def save_augmented_data(image, label, output_image_dir, output_label_dir, prefix):
    """
    将增强后的图像和标签保存到指定目录。
    """
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    image_path = os.path.join(output_image_dir, f"{prefix}.png")
    label_path = os.path.join(output_label_dir, f"{prefix}.png")
    cv2.imwrite(image_path, (image * 255).astype(np.uint8))
    cv2.imwrite(label_path, (label * 255).astype(np.uint8))


# 主函数：对目录中的图像和标签进行增强
def augment_data(image_dir, label_dir, output_image_dir, output_label_dir, num_augmentations=5):
    """
    对指定目录中的图像和标签进行增强，并保存到输出目录。
    """
    # 加载图像和标签对
    image_label_pairs = load_image_label_pairs(image_dir, label_dir)

    # 对每对图像和标签进行增强
    for idx, (image, label) in enumerate(image_label_pairs):
        for aug_idx in range(num_augmentations):
            # 随机选择一种增强方法
            method = random.choice(["geometric", "intensity", "elastic"])  # "cutmix"

            if method == "geometric":
                image_aug, label_aug = geometric_augmentation(image, label)
            elif method == "intensity":
                image_aug = intensity_augmentation(image)
                label_aug = label  # 标签不进行灰度变换
            elif method == "elastic":
                image_aug, label_aug = elastic_deformation(image, label)
            # elif method == "cutmix":
            #     # 随机选择另一对图像和标签进行CutMix
            #     other_image, other_label = random.choice(image_label_pairs)
            #     image_aug, label_aug = cutmix_augmentation(image, label, other_image, other_label)

            # 保存增强后的数据
            prefix = f"pair_{idx}_aug_{aug_idx}"
            save_augmented_data(image_aug, label_aug, output_image_dir, output_label_dir, prefix)


# 示例使用
if __name__ == "__main__":
    # 输入目录（图像和标签）
    image_dir = "/home/server4090/lyx/data/shenzhen_heart/model_data/A3C/train/images"
    label_dir = "/home/server4090/lyx/data/shenzhen_heart/model_data/A3C/train/labels"

    # 输出目录（增强后的图像和标签）
    output_image_dir = "/home/server4090/lyx/data/shenzhen_heart/model_data/A3C/train/images"
    output_label_dir = "/home/server4090/lyx/data/shenzhen_heart/model_data/A3C/train/labels"

    # 对数据进行增强（每对图像和标签生成5个增强样本）
    augment_data(image_dir, label_dir, output_image_dir, output_label_dir, num_augmentations=1)

