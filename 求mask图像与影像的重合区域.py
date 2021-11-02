import os
import cv2
import numpy as np
def image_bitand(src1,src2,dst):
    img = cv2.imread(src1)
    # print(img)
    img1 = cv2.imread(src2)
    # print(img1)
    ronghe = cv2.bitwise_and(img,img1,mask=None,dst=None)
    # print(ronghe)
    # cv2.imshow("image",ronghe)
    # cv2.waitKey(0)
    cv2.imwrite(dst,ronghe)

root_path = r'E:\multimodal ultrasonography\712all_images\214_test_set\ZQM\T\\'
save_path = r'E:\multimodal ultrasonography\712all_images\214_test_data\ZQM_mask\T\\'
images = os.listdir(root_path)
print(len(images))
print(images)
i = 0
while i < len(images):
    if True:
        src1 = root_path + images[i]
        print(src1)
        src2 = root_path + images[i+1]
        print(src2)
        dst =  save_path + images[i].split('.')[0] + 'MASK.jpg'

        print(dst)
        image_bitand(src1,src2,dst)
        i += 4