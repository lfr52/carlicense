import preprocess
import license_letters
import license_province
if __name__ == '__main__':
    # path = "./cut_1/test_img/car2.jpg"
    # car = path.split('/')[-1][:-4]
    # preprocess.find_car_brod(path)
    # preprocess.cut_car_num_for_chart(car)
    license_letters.letter_test("cut_1/img_cut/")
    print('ok')
    license_province.province_test("cut_1/img_cut/")




