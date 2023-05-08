# 裁边
import cv2


def cut_border(img):
    white_row = []  # 记录每一行的白色像素总和
    black_row = []  # ..........黑色.......
    white_col = []  # 记录每一列的白色像素总和
    black_col = []  # ..........黑色.......
    height = img.shape[0]
    width = img.shape[1]

    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img[j][i] == 255:
                s += 1
            if img[j][i] == 0:
                t += 1
        white_col.append(s)
        black_col.append(t)
        # print(i, str(s) + "---------------" + str(t))
    # print("blackmax ---->" + str(max(black_col)) + "------whitemax ------> " + str(max(white_col)))
    arg = True  # False表示白底黑字；True表示黑底白字
    if max(black_col) < max(white_col):
        arg = False

    # 计算每一行的黑白色像素总和
    for i in range(height):
        s = 0  # 这一行白色总数
        t = 0  # 这一行黑色总数
        for j in range(width):
            if img[i][j] == 255:
                s += 1
            if img[i][j] == 0:
                t += 1

        white_row.append(s)
        black_row.append(t)

    w = 0
    if arg:  # 白底的不过循环
        for i in range(width // 2):
            if ((white_col[i + 1] - white_col[i]) if arg else (black_col[i + 1] - black_col[i])) > 25:
                w = i
                break
    j = width
    while j > 0 and arg:  # 白底的不过循环
        if white_col[j - 1] < 10:
            break
        j -= 1

    h = 0
    for i in range(height // 2):
        if ((white_row[i + 1] - white_row[i]) if arg else (black_row[i + 1] - black_row[i])) > 20:
            h = i
            break
    i = height
    while i > 0:
        if white_row[i - 1] < 550:
            break
        i -= 1
    img = img[h:i, w:j]
    img = cv2.resize(img, (width, height))

    return img, arg
