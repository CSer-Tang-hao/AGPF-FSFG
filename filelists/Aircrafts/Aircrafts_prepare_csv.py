import os
import random

from PIL import Image

data_dir = '/home/yuancheng/dataset/fgvc-aircraft-2013b/data'
save_dir = '.'
data_dir = os.path.abspath(data_dir)
save_dir = os.path.abspath(save_dir)

if not os.path.exists(os.path.join(save_dir, 'images')):
    os.makedirs(os.path.join(save_dir, 'images'))

images_dir = os.path.join(data_dir, 'images')
cat_id2name = {}
cat_name2id = {}
with open(os.path.join(data_dir, 'variants.txt')) as f:
    content = f.readlines()
    for i in range(len(content)):
        name = content[i].strip()
        # mkdir(os.path.join(save_dir,str(i)))
        cat_id2name[i] = name
        cat_name2id[name] = i
classes_list = list(cat_name2id.keys())
# exit(0)
img2cat = {}
cat2img = {}
for i in ['images_variant_trainval.txt', 'images_variant_test.txt']:
    with open(os.path.join(data_dir, i)) as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            img = line[:7]
            cat = line[8:]
            cat_id = cat_name2id[cat]
            img2cat[img] = cat_id
            if cat_id not in cat2img:
                cat2img[cat_id] = []
            cat2img[cat_id].append(img)

# support_cat = []
# val_cat = []
# test_cat = []
# # for i in range(100):
#	 if i % 2 == 0:
#		 support_cat.append(i)
#	 elif i % 4 == 1:
#		 val_cat.append(i)
#	 elif i % 4 == 3:
#		 test_cat.append(i)

train_class_num = 60
val_class_num = 15
test_class_num = 25
random.seed(200)
all_list = [i for i in range(100)]
train_list = random.sample(all_list, train_class_num)
remain_list = [rem for rem in all_list if rem not in train_list]
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

    for class_idx in data_list:
        print(dataset, ' ', classes_list[class_idx])
        for img in cat2img[class_idx]:
            img_file = img + '.jpg'
            img = Image.open(os.path.join(images_dir, img_file))
            img = img.convert('RGB')
            img.save(os.path.join(save_dir, 'images', img_file), quality=100)
            file_list.append(os.path.join(save_dir, 'images', img_file))
            label_list.append(class_idx)

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
