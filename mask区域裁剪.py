import os

import cv2
import matplotlib.pyplot as plt

"""
    使用OpenCV截取图片
"""
def search(path):
    left = 1024
    right = 0
    upper = 768
    lower = 0
    img = cv2.imread(path)[:,:,0]
    # print(img.shape)
    for i in range(768):
        for j in range(1024):
            if img[i,j] != 0 :
                # print(img[i,j])
                left = min(j,left)
                right = max(j,right)
                lower = max(i,lower)
                upper =  min(i,upper)
    return (left,upper,right,lower)
def show_cut(path, left, upper, right, lower):
    """
        原图与所截区域相比较
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    """

    img = cv2.imread(path)

    print("This image's size: {}".format(img.shape))  # (H, W, C)

    plt.figure("Image Contrast")
    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(img)  # 展示图片的颜色会改变
    plt.axis('off')

    cropped = img[upper:lower, left:right]

    plt.subplot(1, 2, 2)
    plt.title('roi')
    plt.imshow(cropped)
    plt.axis('off')
    plt.show()


def image_cut_save(path, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    img = cv2.imread(path)  # 打开图像
    cropped = img[upper:lower, left:right]
    # 保存截取的图片
    cv2.imwrite(save_path, cropped)


if __name__ == '__main__':
    root_path = r'E:\multimodal ultrasonography\712all_images\214_test_data\ZQM_mask\C'
    save_path = r'E:\multimodal ultrasonography\712all_images\214_test_data\ZQM_mask\crop_C'
    images = os.listdir(root_path)
    for image in images:
        # print(image)
        pic_path = os.path.join(root_path,image)
        # print(pic_path)
        pic_save_dir_path = os.path.join(save_path,image)
        print(pic_save_dir_path)
        left, upper, right, lower = search(pic_path)
        # show_cut(pic_path, left, upper, right, lower)
        image_cut_save(pic_path, left, upper, right, lower, pic_save_dir_path)

