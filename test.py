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

         

num = 0
l = os.listdir("F:\DeepLearning\lfr/t/")
for rt, dirs, files in os.walk( "F:\DeepLearning\lfr/t/"):

    for filename in files:
        num += 1
print(l, num)
