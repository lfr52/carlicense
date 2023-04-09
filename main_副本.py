'''
车牌框的识别 剪切保存
'''
# 使用的是HyperLPR已经训练好了的分类器
import os
import cv2
from PIL import Image
import time
import numpy as np

from pip._vendor.distlib._backport import shutil

car = 'car3'

def find_car_brod():
    watch_cascade = cv2.CascadeClassifier('./cascade.xml')
    # 先读取图片
    image = cv2.imread(f"./cut_1/test_img/{car}.jpg")
    resize_h = 1000
    height = image.shape[0]
    scale = image.shape[1] / float(image.shape[0])
    image = cv2.resize(image, (int(scale * resize_h), resize_h))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    watches = watch_cascade.detectMultiScale(image_gray, 1.2, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))

    print("检测到车牌数", len(watches))
    for (x, y, w, h) in watches:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cut_img = image[y :y + h, x :x  + w]  # 裁剪坐标为[y0:y1, x0:x1]
        cut_gray = cv2.cvtColor(cut_img, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(f"./cut_1/{car}.jpg", cut_gray)
        im = Image.open(f"./cut_1/{car}.jpg")
        size = 720, 180
        mmm = im.resize(size, Image.ANTIALIAS)
        mmm.save(f"./cut_1/{car}.jpg", "JPEG", quality=90)
        break


'''

剪切后车牌的字符单个拆分保存处理
'''


def cut_car_num_for_chart():
    # 读取图像，并把图像转换为灰度图像并显示
    img = cv2.imread(f"./cut_1/{car}.jpg")  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
    # cv2.imshow('gray', img_gray)  # 显示图片
    # cv2.waitKey(0)

    # 高斯除噪 二值化处理
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow('threshold', th3)
    cv2.imwrite(f'./cut_1/{car}_binary.jpg', th3)
    # cv2.waitKey(0)

    # 分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = th3.shape[0]
    width = th3.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if th3[j][i] == 255:
                s += 1
            if th3[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
        print(i, str(s) + "---------------" + str(t))
    print("blackmax ---->" + str(black_max) + "------whitemax ------> " + str(white_max))
    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True

    n = 1
    start = 1
    end = 2
    temp = 1
    w = 0
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max + 1 if arg else 0.05 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数需多调整，对应下面的0.95
            start = n
            end = find_end(start, white, black, arg, white_max, black_max, width)
            n = end
            # shutil.copy()函数是当检测到这个所谓的 1 时，从样本库中拷贝一张 1 的图片给当前temp下标下的字符
            if end - start > 5 and temp <= 9:   # 移除车牌白条
                print(" end - start" + str(end - start))
                if temp == 1:

                    w = end - start
                    cj = th3[1:height, start:start + w]
                    cv2.imwrite(f"./cut_1\img_cut_initial\\{car}_" + str(temp) + ".jpg", cj)

                    im = Image.open(f"./cut_1\img_cut_initial\\{car}_" + str(temp) + ".jpg")
                    size = 32, 40
                    mmm = im.resize(size, Image.ANTIALIAS)
                    mmm.save(f"./cut_1\img_cut\\{car}_" + str(temp) + ".jpg", quality=95)
                    temp = temp + 1
                    pass
                elif temp == 3:
                    
                    cj = th3[1:height, start:end]
                    cv2.imwrite(f"./cut_1\img_cut_initial\\{car}_" + str(temp) + ".jpg", cj)

                    im = Image.open(f"./cut_1\img_cut_initial\\{car}_" + str(temp) + ".jpg")
                    size = 32, 40
                    mmm = im.resize(size, Image.ANTIALIAS)
                    mmm.save(f"./cut_1\img_cut\\{car}_" + str(temp) + ".bmp", quality=95)
                    temp = temp + 1
                    pass
                else:
                    cj = th3[1:height, start:start + w]
                    cv2.imwrite(f"./cut_1\img_cut_initial\\{car}_" + str(temp) + ".jpg", cj)

                    im = Image.open(f"./cut_1\img_cut_initial\\{car}_" + str(temp) + ".jpg")
                    size = 32, 40
                    mmm = im.resize(size, Image.ANTIALIAS)
                    mmm.save(f"./cut_1\img_cut\\{car}_" + str(temp) + ".jpg", quality=95)
                    temp = temp + 1


# 分割图像
def find_end(start_, white, black, arg, white_max, black_max, width):
    end_ = start_ + 1
    for m in range(start_ + 1, width -1):
        if m == width -2:
            return width -1
        if (black[m] if arg else white[m]) > (0.95 * black_max - 2 if arg else 0.95 * white_max):  # 0.95这个参数需多调整，对应上面的0.05
            end_ = m
            break
    return end_


# find_car_brod()   #车牌定位裁剪

cut_car_num_for_chart() #二值化处理裁剪成单个字符
