import time
import cv2

import license_letters
# import license_province
import preprocess_2
# open
cap = cv2.VideoCapture(1)
flag = cap.isOpened()
print(cap.isOpened(), cap.get(3), cap.get(4))
count = 1
while flag:
    isTrue, image = cap.read()
    # frame = cv2.resize(image, (800, 600))

    if isTrue:
        cv2.imshow('My Video', image)
        begin = time.time()
        watch_cascade = cv2.CascadeClassifier('./cascade.xml')
        # 先读取图片
        resize_h = 1000
        height = image.shape[0]
        scale = image.shape[1] / float(image.shape[0])
        image = cv2.resize(image, (int(scale * resize_h), resize_h))
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        watches = watch_cascade.detectMultiScale(image_gray, 1.2, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))
        if len(watches) == 0:
            pass
            # print('no license')
        else:
            watch = watches[0]
            x, y, w, h = watch
            cv2.imwrite(f'video/{count}_complete.jpg', image)
            cut_img = image[y:y + h + 3, x:x + w + 10]  # 2
            cut_gray = cv2.cvtColor(cut_img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(cut_gray, (720, 180))
            cv2.imshow('license', img)
            times = time.time() - begin
            print('检测时间%.4f' % times)
            if cv2.waitKey() == ord('c'):
                continue
            if cv2.waitKey() == ord('r'):
                cv2.imwrite(f'video/{count}.jpg', img)
                print('test')
                time_begin = time.time()
                preprocess_2.cut_car_num_for_chart(img, count)
                # license_province.province_test("video/img_cut/")
                license_letters.letter_test("video/img_cut/", count)
                count = count + 1
                time_total = time.time() - time_begin
                print('测试总时间为%.4f秒' % time_total)

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
