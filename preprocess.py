'''
车牌框的识别 剪切保存
'''
# 使用的是HyperLPR已经训练好了的分类器
import os
import cv2
from PIL import Image
import numpy as np

import cut_border
import rotate


# 找到车牌并提取，保存
def find_car_brod(path):
    watch_cascade = cv2.CascadeClassifier('./cascade.xml')
    # 先读取图片
    image = cv2.imread(path)
    resize_h = 1000
    height = image.shape[0]
    scale = image.shape[1] / float(image.shape[0])
    image = cv2.resize(image, (int(scale * resize_h), resize_h))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    watches = watch_cascade.detectMultiScale(image_gray, 1.2, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))

    watch = watches[0]

    x, y, w, h = watch  # [ 140  281 1131  288]
    print(watch)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # cut_img = image[y + 3:y + h - 20, x:x + w - 20]  # 1
    cut_img = image[y + 3:y + h - 3, x:x + w + 10]  # 裁剪坐标为[y0:y1, x0:x1]  2
    # cut_img = image[y + 3:y + h - 3, x:x + w - 4]  # 3
    # cut_img = image[y + 3:y + h - 3, x - 50:x + w + 10] # 4
    cut_gray = cv2.cvtColor(cut_img, cv2.COLOR_RGB2GRAY)

    save_path = path.split('/')
    save_path = os.path.join(save_path[0], save_path[1], save_path[3])
    save_path = save_path.replace('\\', '/')  #

    cv2.imwrite(save_path, cut_gray)
    im = Image.open(save_path)
    size = 720, 180
    mmm = im.resize(size, Image.ANTIALIAS)
    mmm.save(save_path, "JPEG", quality=90)


# 拆分字符
def cut_car_num_for_chart(car):
    # 读取图像，并把图像转换为灰度图像并显示
    img = cv2.imread(f"./cut_1/{car}.jpg")  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
    # cv2.imshow('gray', img_gray)  # 显示图片
    # cv2.waitKey(0)

    # 高斯除噪 二值化处理
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow('threshold', th3)
    # cv2.waitKey(0)
    th3, arg = cut_border.cut_border(th3)
    cv2.imwrite(f'./cut_1/{car}_binary.jpg', th3)
    # cv2.waitKey(0)
    #

    th3 = rotate.auto_rotate(th3)
    cv2.imwrite(f'./cut_1/{car}_binary_r.jpg', th3)

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

    n = 0
    start = 1
    end = 2
    temp = 1
    w = 0
    start_4 = 0
    space = 0
    while n < width - 1:
        n += 1
        if temp == 4:
            break
        if (white[n] if arg else black[n]) > (0.05 * white_max + 1 if arg else 0.05 * black_max):
            # 用来判断是白底黑字还是黑底白字
            # 0.05这个参数需多调整，对应下面的0.95
            start = n
            end = find_end(start, white, black, arg, white_max, black_max, width)
            # if temp > 3:
            #     n = max(end, n + w - 1)
            # else:
            n = end
            if end - start > 15:  # 移除车牌白条
                print(" end - start" + str(end - start))
                print(start, end, n)
                if temp == 1:  # 汉字
                    cj = th3[1:height, start:end + 2]
                    space = end + 2
                    save_pic(cj, temp, car)
                    temp = temp + 1
                    pass
                else:
                    if temp == 2:
                        space = start - space
                        w = end - start + 2  # 经验值
                    start_cj, end_cj = val_pic(th3, start, start + w, arg)
                    cj = th3[1:height, start_cj:end_cj]
                    start_4 = start + w
                    save_pic(cj, temp, car)
                    temp = temp + 1

    # space = 0
    start_4 = int(start_4 + space)
    while temp < 9:
        end_4 = min(start_4 + w, width)
        if end_4 - start_4 > w / 2:
            start_4, end_4 = val_pic(th3, start_4, end_4, arg)
            cj = th3[1:height, start_4:end_4]
            save_pic(cj, temp, car)
        start_4 = end_4 + space
        temp = temp + 1


# 保存单个字符图片
def save_pic(img, num, car):
    cv2.imwrite(f"./cut_1/img_cut_initial/{car}_{str(num)}.jpg", img)
    im = Image.open(f"./cut_1/img_cut_initial/{car}_{str(num)}.jpg")
    size = 32, 40
    mmm = im.resize(size, Image.ANTIALIAS)
    mmm.save(f"./cut_1/img_cut/{car}_{str(num)}.jpg", quality=95)


# 找到end
def find_end(start_, white, black, arg, white_max, black_max, width):
    end_ = start_ + 1
    for m in range(start_ + 1, width - 1):
        if m == width - 2:
            return width - 1
        if (black[m] if arg else white[m]) > (
                0.95 * black_max - 2 if arg else 0.95 * white_max):  # 0.95这个参数需多调整，对应上面的0.05
            end_ = m
            break
    return end_


def val_pic(img, start, end, arg):
    height = img.shape[0]
    width = img.shape[1]

    for i in range(end, width):
        flag = 0
        for j in range(height):
            if img[j][i] == 255 if arg else img[j][i] == 0:
                flag = 1
        if flag == 0:  # 0为这一列全黑
            end = i
            break
        if i == width - 1:
            end = i

    for k in range(start, 0, -1):
        flag = 0
        for j in range(height):
            if img[j][k] == 255 if arg else img[j][k] == 0:
                flag = 1
        if flag == 0:  # 0为这一列全黑
            start = k
            break
    return start, end


if __name__ == '__main__':
    path = "./cut_1/test_img/car2.jpg"
    car = path.split('/')[-1][:-4]
    find_car_brod(path)
    cut_car_num_for_chart(car)
