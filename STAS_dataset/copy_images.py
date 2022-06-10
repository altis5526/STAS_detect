import glob
import shutil
import os

src_dir = "training/Train_Images"
label_src_dir = "training/Train_Annotations"
dst_dir = "Test_Images"
label_dst_dir = "valid0_labels"

copy_files = []
copy_labels = []

with open("valid0.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        copy_files.append(line)
        copy_labels.append(line)

for i in range(len(copy_files)):
    copy_files[i] = copy_files[i][:-1] + '.jpg'
    copy_labels[i] = copy_labels[i][:-1] + '.txt'

# for images in copy_files:
#     image_path = os.path.join(src_dir, images)
#     shutil.copy(image_path, dst_dir)

for label in copy_labels:
    label_path = os.path.join(label_src_dir, label)
    print(label_path)
    shutil.copy(label_path, label_dst_dir)