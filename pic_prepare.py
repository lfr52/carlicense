import os
import numpy as np
import cv2

def reverse(pic):
    x0, x1, x2, x3 = pic[0][0], pic[0][31], pic[39][0], pic[39][31]
    x = int(x0 >= 253) + int(x1 >= 253) + int(x2 >= 253) + int(x3 >= 253)
    if x >= 3:
        for i in range(len(pic)):
            for j in range(len(pic[i])):
                if pic[i][j] > 127:
                    pic[i][j] = 0
                else:
                    pic[i][j] = 255
    return pic

def pic_count(dir, num_class):
    count = 0
    for i in range(0,num_class):
        dir_1 = dir + '%s/' % i # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir_1):
            for filename in files:
                count += 1
    return count

def pic_intput(path, input_image, size):
    pic_org = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    ret, pic_new = cv2.threshold(src=pic_org,             # 要二值化的图片
                thresh=0,                           # 全局阈值
                maxval=255,                           # 大于全局阈值后设定的值
                type=cv2.THRESH_BINARY|cv2.THRESH_OTSU)               # 设定的二值化类型，
    pic_new = reverse(pic_new)
    pic_new = pic_new.reshape([1,size])
    # print(pic_new)
    for j in range(size):
        if pic_new[0][j] > 127:
            input_image[j] = 1
    # print(input_image)
    return input_image


if __name__ =='__main__':
    # # result = pic_count('data/vaildation_set/', 37)
    # # print(result)
    # input_count = pic_count('data/vaildation_set/', 37)

    # # 定义对应维数和各维长度的数组
    # input_images = np.array([[0]*1280 for i in range(input_count)])
    # input_labels = np.array([[0]*37 for i in range(input_count)])


    # index = 0
    # for i in range(0,37):
    #     dir = 'data/vaildation_set/%s/' % i # 这里可以改成你自己的图片目录，i为分类标签
    #     for rt, dirs, files in os.walk(dir):
    #         for filename in files:
    #             input_images[index] = pic_intput(dir + filename, input_images[index], 1280)

    #             input_labels[index][i] = 1
    #             index += 1
    # print(input_images)  
    # print(len(input_images), len(input_images[0]))
    # print(input_labels)
    # print(len(input_labels), len(input_labels[0]))

    # f = 'F:/DeepLearning/lfr/data_binary_otsu/training_set/5/2_1501494057_504226.bmp'
    # img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # img2 = copy.deepcopy(img)
    # img2 = reverse(img2)
    # cv2.namedWindow('before', 0)
    # cv2.namedWindow('after', 0)
    # cv2.imshow('before', img)
    # cv2.imshow('after', img2)
    # cv2.waitKey()

    for i in range(0,37):
        dir = 'data_binary_otsu/vaildation_set/%s/' % i # i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                pic = cv2.imdecode(np.fromfile(dir + filename, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                pic = reverse(pic)
                pic_new_path = 'data_binary_otsu2/vaildation_set/%s/' % i
                if not os.path.exists(pic_new_path):
                    os.makedirs(pic_new_path)  
                cv2.imencode(filename[-4:], pic)[1].tofile(pic_new_path + filename) 

