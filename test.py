# PROVINCES = ("京","闽","粤","苏","沪","浙")
# print(PROVINCES)
# print(type(PROVINCES))

# p=[]

# num = 37
# file = open('./车牌字符顺序.txt', encoding='utf-8')
# for n in range(0, num):
#     a = file.readline()
#     # print(type(a))
#     # print(a)
#     # print(a[-2:-1])
#     p.append(a[-2:-1])
# print(p)
# file.close()    

# print(p[1])
# print(len(p))  #37


# result=[10, 9 , 11,0 ,100,1,2]
# max1 = 0
# max2 = 0
# max3 = 0
# max1_index = 0
# max2_index = 0
# max3_index = 0
# for j in range(7):
#     if result[j] > max1:
#         max3 = max2
#         max3_index = max2_index
#         max2 = max1
#         max2_index = max1_index
#         max1 = result[j]
#         max1_index = j
#         continue
#     if (result[j]>max2) :#and (result[j]<=max1)
#         max3 = max2
#         max3_index = max2_index
#         max2 = result[j]
#         max2_index = j
#         continue
#     if (result[j]>max3) :#and (result[j]<=max2)
#         max3 = result[j]
#         max3_index = j
#         continue
# print("%s %s\n" %(max1 , max1_index))
# print("%s %s\n" %(max2 , max2_index))
# print("%s %s\n" %(max3 , max3_index))


from PIL import Image
import os
import cv2
import numpy as np
import copy
# width, height = 32, 40     
# for i in range(0,37):
#     dir = 'data/vaildation_set0/%s/' % i # 这里可以改成你自己的图片目录，i为分类标签
#     for rt, dirs, files in os.walk(dir):
#         for filename in files:
#             filename1 = dir + filename
#             print(filename1)
#             pic_org = Image.open(filename1)
#             pic_new = pic_org.resize((width, height), Image.ANTIALIAS)
#             pic_new_path = 'data/vaildation_set/%s/' % i
#             if not os.path.exists(pic_new_path):
#                 os.makedirs(pic_new_path)  
#             pic_new.save(pic_new_path + filename)


# name = '572.bmp'
# # img = Image.open('F:/DeepLearning/lfr/data/training_set/1'+name)
# path= 'F:/DeepLearning/lfr/data/training_set/2/'+name
# print(path)
# image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # cv2.imread(path)
# cv2.namedWindow('before', 0)
# cv2.namedWindow('after', 0)
# cv2.imshow('before', image)
# ret, new = cv2.threshold(src=image,             # 要二值化的图片
#                        thresh=127,              # 全局阈值
#                        maxval=255,              # 大于全局阈值后设定的值
#                        type=cv2.THRESH_BINARY)  # 设定的二值化类型，
# cv2.imshow('after', new)
# cv2.waitKey()


# for i in range(0,37):
#     dir = 'data/vaildation_set/%s/' % i # i为分类标签
#     for rt, dirs, files in os.walk(dir):
#         for filename in files:
#             pic_org = cv2.imdecode(np.fromfile(dir + filename, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
#             ret, pic_new = cv2.threshold(src=pic_org,             # 要二值化的图片
#                         thresh=0,                           # 全局阈值
#                         maxval=255,                           # 大于全局阈值后设定的值
#                         type=cv2.THRESH_BINARY|cv2.THRESH_OTSU)               # 设定的二值化类型，
#             pic_new_path = 'data_binary_otsu/vaildation_set/%s/' % i
#             if not os.path.exists(pic_new_path):
#                 os.makedirs(pic_new_path)  
#             cv2.imencode(filename[-4:], pic_new)[1].tofile(pic_new_path + filename) 


# f = 'F:/DeepLearning/lfr/data_binary_otsu/training_set/5/1_4466000000002271219313-1.jpg'
# img = Image.open(f)
# width = img.size[0]
# height = img.size[1]
# print(width)  #32
# print(height) #40
# input_images = np.array([[0]*1280 for i in range(2)])
# print(input_images) 
# np.savetxt( "a.csv", input_images, delimiter="," )
# for h in range(0, height):
#     for w in range(0, width):
#                         # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
#         if img.getpixel((w, h)) > 127:
#             input_images[0][w+h*width] = 0
#         else:
#             input_images[0][w+h*width] = 1
# np.savetxt( "a0.csv", input_images, delimiter="," )


# a, b = 0 , 0
# img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
# print(img[0][0], img[0][31], img[39][0], img[39][31])
# img2 = copy.deepcopy(img)


# img2[0][0], img2[0][31], img2[39][0], img2[39][31] = 0, 0, 0, 0
# cv2.namedWindow('before', 0)
# cv2.namedWindow('after', 0)
# cv2.imshow('before', img)
# print(img2)
# print(img)
# cv2.imshow('after', img2)
# cv2.waitKey()
# img = img.reshape([1,1280])
# for i in range(1280):
#     if(img[0][i] == 0):
#         a = a + 1
#     if(img[0][i] == 255):
#         b = b + 1
# print(a,b)


# num = 0
# l = os.listdir("F:\DeepLearning\lfr/t/")
# for rt, dirs, files in os.walk( "F:\DeepLearning\lfr/t/"):
#
#     for filename in files:
#         num += 1
# print(l, num)


#
# import math
# from typing import Optional, List
#
# import torch
# from torch import nn
#
#
# class PrepareForMultiHeadAttention(nn.Module):
#
#
#     def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
#         super().__init__()
#         # Linear layer for linear transform
#         self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
#         # Number of heads
#         self.heads = heads
#         # Number of dimensions in vectors in each head
#         self.d_k = d_k
#
#     def forward(self, x: torch.Tensor):
#         # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
#         # We apply the linear transformation to the last dimension and split that into
#         # the heads.
#         head_shape = x.shape[:-1]
#
#         # Linear transform
#         x = self.linear(x)
#
#         # Split last dimension into heads
#         x = x.view(*head_shape, self.heads, self.d_k)
#
#         # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
#         return x
#
#
# class MultiHeadAttention(nn.Module):
#     r"""
#     <a id="MHA"></a>
#     ## Multi-Head Attention Module
#     This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.
#     $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
#     In simple terms, it finds keys that matches the query, and gets the values of
#      those keys.
#     It uses dot-product of query and key as the indicator of how matching they are.
#     Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
#     This is done to avoid large dot-product values causing softmax to
#     give very small gradients when $d_k$ is large.
#     Softmax is calculated along the axis of of the sequence (or time).
#     """
#
#     def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
#         """
#         * `heads` is the number of heads.
#         * `d_model` is the number of features in the `query`, `key` and `value` vectors.
#         """
#
#         super().__init__()
#
#         # Number of features per head
#         self.d_k = d_model // heads
#         # Number of heads
#         self.heads = heads
#
#         # These transform the `query`, `key` and `value` vectors for multi-headed attention.
#         self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
#         self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
#         self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
#
#         # Softmax for attention along the time dimension of `key`
#         self.softmax = nn.Softmax(dim=1)
#
#         # Output layer
#         self.output = nn.Linear(d_model, d_model)
#         # Dropout
#         self.dropout = nn.Dropout(dropout_prob)
#         # Scaling factor before the softmax
#         self.scale = 1 / math.sqrt(self.d_k)
#
#         # We store attentions so that it can be used for logging, or other computations if needed
#         self.attn = None
#
#     def get_scores(self, query: torch.Tensor, key: torch.Tensor):
#         """
#         ### Calculate scores between queries and keys
#         This method can be overridden for other variations like relative attention.
#         """
#
#         # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
#         return torch.einsum('ibhd,jbhd->ijbh', query, key)
#
#     def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
#         """
#         `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
#         If the query dimension is equal to $1$ it will be broadcasted.
#         """
#
#         assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
#         assert mask.shape[1] == key_shape[0]
#         assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
#
#         # Same mask applied to all heads.
#         mask = mask.unsqueeze(-1)
#
#         # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
#         return mask
#
#     def forward(self, *,
#                 query: torch.Tensor,
#                 key: torch.Tensor,
#                 value: torch.Tensor,
#                 mask: Optional[torch.Tensor] = None):
#         """
#         `query`, `key` and `value` are the tensors that store
#         collection of *query*, *key* and *value* vectors.
#         They have shape `[seq_len, batch_size, d_model]`.
#         `mask` has shape `[seq_len, seq_len, batch_size]` and
#         `mask[i, j, b]` indicates whether for batch `b`,
#         query at position `i` has access to key-value at position `j`.
#         """
#
#         # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
#         seq_len, batch_size, _ = query.shape
#
#         if mask is not None:
#             mask = self.prepare_mask(mask, query.shape, key.shape)
#
#         # Prepare `query`, `key` and `value` for attention computation.
#         # These will then have shape `[seq_len, batch_size, heads, d_k]`.
#         query = self.query(query)
#         key = self.key(key)
#         value = self.value(value)
#
#         # Compute attention scores $Q K^\top$.
#         # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
#         scores = self.get_scores(query, key)
#
#         # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
#         scores *= self.scale
#
#         # Apply mask
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#
#         # $softmax$ attention along the key sequence dimension
#         # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
#         attn = self.softmax(scores)
#
#         # Save attentions if debugging
#         # tracker.debug('attn', attn)
#
#         # Apply dropout
#         attn = self.dropout(attn)
#
#         # Multiply by values
#         # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
#         x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
#
#         # Save attentions for any other calculations
#         self.attn = attn.detach()
#
#         # Concatenate multiple heads
#         x = x.reshape(seq_len, batch_size, -1)
#
#         # Output layer
#         return self.output(x)
#
# if __name__ == '__main__':
#     multiHeadAttention = MultiHeadAttention(6, 18)
#     value = torch.ones(2, 3, 18)
#     key = torch.rand(2, 3, 18)
#     query = torch.rand(2, 3, 18)
#     out = multiHeadAttention.forward(query=query, key=key, value=value)
#     print('ok')


# import cv2
# img = cv2.imread('F:\DeepLearning\lfr\\t\\car3_6.jpg')
# image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# print('ok')
#
#
# fname_list = os.listdir("F:\DeepLearning\lfr/t2/")
# num = len(fname_list)
#
# for n in range(0, num):
#     print(fname_list[n])
#     path = "F:\DeepLearning\lfr\\t2/" + fname_list[n]
#     img = cv2.imread(path)
#     img = cv2.resize(img, (32, 40))
#     cv2.imwrite(path, img)

import cv2
from random import *
import math


# def auto_rotate(path, angle):
#     cv2.namedWindow('initial', 0)
#     cv2.namedWindow('rotated', 0)
#     cv2.namedWindow('out', 0)
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('initial', img)
#     img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#
#     h, w = img.shape[:2]
#     center = (w//2, h//2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#
#     rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     cv2.imshow('rotated', rotated)
#
#     # rep = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     # cv2.imshow('after', rep)
#     # cv2.waitKey(0)
#     # ele = np.where(rotated > 254)
#     # coords = np.column_stack(np.where(rotated > 0))
#     # print(coords)
#     imgCanny = cv2.Canny(rotated, 50, 50)#canny边缘检测
#     contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#提取图像外轮廓，并返回至contours
#     cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]#所有的排序，然后取得最大的轮廓
#
#     rect = cv2.minAreaRect(cnts)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     img_output = cv2.drawContours(cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR),  [box], -1, (0, 255, 0), 3)
#     cv2.imshow('rotated', img_output)
#
#     angle = rect[-1]
#     print(angle)
#     M2 = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_2 = cv2.warpAffine(rotated, M2, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     out = rotated_2[100:-100, 100:-100]
#     _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow('out', out)
#     cv2.waitKey(0)
#     print('ok')

# v1
# def auto_rotate(path, path2):
#     cv2.namedWindow('initial', 0)
#     cv2.namedWindow('rotated', 0)
#     cv2.namedWindow('out', 0)
#     cv2.namedWindow('canny', 0)
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('initial', img)
#     img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#
#     h, w = img.shape[:2]
#     center = (w // 2, h // 2)
#
#     imgCanny = cv2.Canny(img, 50, 50)  # canny边缘检测
#     cv2.imshow('canny', imgCanny)
#     contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 提取图像外轮廓，并返回至contours
#     cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 所有的排序，然后取得最大的轮廓
#
#     rect = cv2.minAreaRect(cnts)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     img_output = cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),  [box], -1, (0, 255, 0), 3)
#     cv2.imshow('rotated', img_output)
#
#     angle = rect[-1]
#     print(angle)
#     if angle > 45:
#         angle = -(90 - angle)
#     else:
#         angle = - angle
#     m = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     out = rotated[100:-100, 100:-100]
#     _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow('out', out)
#     cv2.waitKey(0)
#     cv2.imwrite(path2, out)
#     print('ok')

# v2
def auto_rotate(path, path2):
    cv2.namedWindow('initial', 0)
    cv2.namedWindow('rotated', 0)
    cv2.namedWindow('out', 0)
    cv2.namedWindow('canny', 0)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('initial', img)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    imgCanny = cv2.Canny(img, 50, 50)  # canny边缘检测
    cv2.imshow('canny', imgCanny)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 提取图像外轮廓，并返回至contours
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

    w = w - 200
    h = h - 200
    a = math.fabs(angle) / 180 * math.pi
    w_new = (w * math.cos(a) - h * math.sin(a)) / (math.cos(a) ** 2 - math.sin(a) ** 2)
    h_new = (h * math.cos(a) - w * math.sin(a)) / (math.cos(a) ** 2 - math.sin(a) ** 2)

    out = rotated[int(100 + h // 2 - h_new // 2):int(100 + h // 2 + h_new // 2),
                  int(100 + w // 2 - w_new // 2):int(100 + w // 2 + w_new // 2)]

    _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('out', out)
    cv2.waitKey(0)
    cv2.imwrite(path2, out)
    print('ok')


def rotate(path, path2, angle):
    cv2.namedWindow('initial', 0)
    cv2.namedWindow('rotated', 0)
    cv2.namedWindow('out', 0)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('initial', img)
    img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC)
    # rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow('rotated', rotated)
    w = w - 400
    h = h - 400
    a = angle / 180 * math.pi
    w_new = w * math.cos(a) + h * math.sin(a)
    h_new = w * math.sin(a) + h * math.cos(a)
    i, j = 200 + h // 2 - h_new // 2, 200 + h // 2 + h_new // 2
    out = rotated[int(200 + h // 2 - h_new // 2):int(200 + h // 2 + h_new // 2),
                  int(200 + w // 2 - w_new // 2):int(200 + w // 2 + w_new // 2)]
    _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('out', out)
    cv2.imwrite(path2, out)
    cv2.waitKey(0)


if __name__ == '__main__':
    # path = 'cut_1/img_cut_initial_423/'
    # fname_list = os.listdir(path)
    # for i in fname_list:
    #     pic_path = path + i
    #     print(pic_path)
    #     save_dir = 'cut_1/rotated/' + i
    #     a = randint(0, 45)
    #     print(a)
    #     rotate(pic_path, save_dir, a)

    # path = 'cut_1/rotated/'
    # # path = 'cut_1/img_cut_initial_423/'
    # fname_list = os.listdir(path)
    # for i in fname_list:
    #     pic_path = path + i
    #     print(pic_path)
    #     save_dir = 'cut_1/auto_rotated/' + i
    #     # save_dir = 'cut_1/1/' + i
    #     auto_rotate(pic_path, save_dir)

    path = 'cut_1/car1_binary_r20.jpg'
    save_dir = 'cut_1/car1_binary_rr20.jpg'
    auto_rotate(path, save_dir)

    # path = 'cut_1/car1_binary.jpg'
    # save_dir = 'cut_1/car1_binary_r20.jpg'
    # rotate(path, save_dir, 20)
