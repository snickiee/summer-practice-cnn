import cv2

from glob import glob
from sklearn.model_selection import train_test_split

import os

#параметры

src_dir = r"D:\Desktop\product-classification-master\data\product_category\dataset_augmented"
root_dir = r"D:\Desktop\product-classification-master\data\product_category\dataset_augmented"

class_list_filename = r"D:\Desktop\product-classification-master\data\product_category\freiburg_groceries_classList.txt"


p_train = 0.7
p_val = 0.15
p_test = 0.15

#функции
def create_class_list(list_filename):

    with open(list_filename, 'r') as f:

        list = []
        counter = 0

        for line in f.readlines():

            category, newLine = line.strip().split(',')
            list.append(category)
            counter+=1

    return list

def create_trainval_batch(class_name):

    img_array=[]

    print("\nКласс: %s" % class_name)

    img_orig_folder_name = src_dir+ '\\' + class_name
    img_counter = 0

    for fn in glob(img_orig_folder_name+'/*.'+'png'):

        img_cv = cv2.imread(fn)

        img_array.append(img_cv)
        img_counter+=1

    print("%s: %i картинок" %(img_orig_folder_name, img_counter))


#разделение датасета

    #разделение в train/val/test 
    x_train,x_remain  = train_test_split(img_array, shuffle=True, test_size=p_val+p_test, random_state=0)
    x_val, x_test = train_test_split(x_remain, shuffle=True, test_size=p_test/(p_val+p_test) , random_state=0)

    print("x_train:"+str(len(x_train)))
    print("x_val:"+str(len(x_val)))
    print("x_test:"+str(len(x_test)))

    # куда сохранять
    save_train = root_dir + 'train\\' + class_name + '\\'
    save_val = root_dir + 'val\\' + class_name + '\\'
    save_test = root_dir + 'test\\' + class_name + '\\'

    save_images(save_train,class_name,x_train)
    save_images(save_val, class_name, x_val)
    save_images(save_test, class_name, x_test)


def save_images(save_dir, class_name, x_set):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, img_set in enumerate(x_set):
        if i < 10:
            padding = '000'
        elif i >= 10 and i < 100:
            padding = '00'
        else:
            padding = '0'

        filename_save = class_name + padding + str(i) + '.png'
        try:
            cv2.imwrite(filename=os.path.join(save_dir, filename_save), img=img_set)
        except Exception as e:
            print(f"Ошибка {filename_save}: {e}")
        else:
            print(f"Сохраняем {filename_save} в {save_dir}")

    print(f"Сохраняем {len(x_set)} изображения в {save_dir}")


#main
if __name__=="__main__":

    class_list = create_class_list(class_list_filename)

    for category in class_list:
        create_trainval_batch(category)
