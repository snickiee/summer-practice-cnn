import cv2
import numpy as np
import scipy

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from glob import glob

from keras.src.legacy.preprocessing.image import ImageDataGenerator


#параметры
root_dir = r"D:\Desktop\product-classification-master\data\product_category\original"
class_list_filename = r"D:\Desktop\product-classification-master\data\product_category\freiburg_groceries_classList.txt"

imgs_per_class = 400

# функции
def create_class_list(list_filename):

    with open(list_filename, 'r') as f:
        list = []

        counter = 0
        for line in f.readlines():
            category, newLine = line.strip().split(',')
            list.append(category)
            counter+=1

    return list

def create_img_batch(folder_name):

    img_array=[]

    counter=0
    for fn in glob(folder_name+'/*.'+'png'):

        img_cv = cv2.imread(fn)
        img_cv = cv2.resize(img_cv,(224,224),img_cv)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        img_array.append(img_cv)
        counter+=1

    batch = np.array(img_array)

    return batch

#main

class_list = create_class_list(class_list_filename)

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range= 0.1, shear_range=0.05)

# открытие каждой папки класса
for category in class_list:
    class_dir = root_dir + '\\' + category
    print(class_dir)
    x_batch = create_img_batch(class_dir)

    new_img_counter = 0
    n_img_max = imgs_per_class-x_batch.shape[0]

    print("\nКласс: %s" %category)
    print("Кол-во картинок: %i " %x_batch.shape[0])

    # генерировать, пока в сумме не выйдет 400
    for batch in datagen.flow(x=x_batch,batch_size=1,save_to_dir=class_dir,save_prefix=category):
        new_img_counter+=1

        if(new_img_counter >= n_img_max):
            break

    print("Создано %i новых картинок " %new_img_counter)

'''******************************************************************************************************************'''