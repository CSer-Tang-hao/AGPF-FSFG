##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Wenbin Li
## Date: Dec. 16 2018
##
## Divide data into train/val/test in a csv version
## Output: train.csv, val.csv, test.csv 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import random

from PIL import Image

data_dir = '/home/yuancheng/dataset/dogs'  # the path of the download dataset
save_dir = '.'  # the saving path of the divided dataset

data_dir = os.path.abspath(data_dir)
save_dir = os.path.abspath(save_dir)

if not os.path.exists(os.path.join(save_dir, 'images')):
    os.makedirs(os.path.join(save_dir, 'images'))

images_dir = os.path.join(data_dir, 'Images')
train_class_num = 70
val_class_num = 20
test_class_num = 30

# get all the dog classes
classes_list = [class_name for class_name in os.listdir(images_dir) if
                os.path.isdir(os.path.join(images_dir, class_name))]
class_to_idx = {c: i for (i, c) in enumerate(classes_list)}

# divide the train/val/test set
random.seed(120)
train_list = random.sample(classes_list, train_class_num)
remain_list = [rem for rem in classes_list if rem not in train_list]
val_list = random.sample(remain_list, val_class_num)
test_list = [rem for rem in remain_list if rem not in val_list]

dataset_list = ['base', 'val', 'novel']
for dataset in dataset_list:
    file_list = []
    label_list = []
    data_list = None
    if dataset == 'base':
        data_list = train_list
    if dataset == 'val':
        data_list = val_list
    if dataset == 'novel':
        data_list = test_list

    for class_name in data_list:
        print(dataset, ' ', class_name)
        images = [i for i in os.listdir(os.path.join(images_dir, class_name))]
        for index, img_file in enumerate(images):
            img = Image.open(os.path.join(images_dir, class_name, img_file))
            img = img.convert('RGB')

            img.save(os.path.join(save_dir, 'images', img_file), quality=100)
            file_list.append(os.path.join(save_dir, 'images', img_file))
            label_list.append(class_to_idx[class_name])

    fo = open(os.path.join(save_dir, dataset + ".json"), "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in classes_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" % dataset)
