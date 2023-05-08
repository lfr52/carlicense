import cv2
import numpy as np


def auto_rotate(img):
    cv2.namedWindow('initial', 0)
    cv2.namedWindow('rotated', 0)
    cv2.namedWindow('out', 0)
    cv2.namedWindow('canny', 0)
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('initial', img)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    imgcanny = cv2.Canny(img, 50, 50)  # canny边缘检测
    cv2.imshow('canny', imgcanny)
    contours, hierarchy = cv2.findContours(imgcanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 提取图像外轮廓，并返回至contours
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 所有的排序，然后取得最大的轮廓

    rect = cv2.minAreaRect(cnts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img_output = cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [box], -1, (0, 255, 0), 3)
    cv2.imshow('rotated', img_output)

    angle = rect[-1]
    print(angle)
    if angle > 45:
        angle = -(90 - angle)
    else:
        angle = - angle
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out = rotated[100:-100, 100:-100]
    _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('out', out)
    # cv2.waitKey(0)
    # cv2.imwrite(path2, out)
    print('ok')
    return out
