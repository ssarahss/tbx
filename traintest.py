import os
import shutil

def move_file(source_path, destination_path):
    shutil.move(source_path, destination_path)

# Path folder sumber
source_folder = r'C:\TBX11K\imgs'

# Path file txt yang berisi daftar file untuk data latih (train)
train_txt_file = r'C:\TBX11K\lists\TBX11K_train.txt'

# Path file txt yang berisi daftar file untuk data validasi (val)
val_txt_file = r'C:\TBX11K\lists\TBX11K_val.txt'

# Path folder tujuan untuk data latih (train)
train_destination_folder = r'C:\TBX11K\train'

# Path folder tujuan untuk data validasi (val)
val_destination_folder = r'C:\TBX11K\val'

# Membuat folder tujuan untuk data latih dan validasi
os.makedirs(train_destination_folder, exist_ok=True)
os.makedirs(val_destination_folder, exist_ok=True)

# label folder dan nilai sebagai nama folder tujuan
label_dict = {
    'health': 'health',
    'sick': 'sick',
    'tb': 'tb'
}

# Memindahkan file-file untuk data latih (train)
with open(train_txt_file, 'r') as train_file:
    for line in train_file:
        file_path = line.strip()
        file_label = file_path.split('/')[-2]
        folder_name = label_dict[file_label]
        file_path = os.path.join(r'C:\TBX11K\imgs', file_path)
        destination_path = os.path.join(train_destination_folder, folder_name)
        os.makedirs(destination_path, exist_ok=True)
        move_file(file_path, destination_path)

# Memindahkan file-file untuk data validasi (val)
with open(val_txt_file, 'r') as val_file:
    for line in val_file:
        file_path = line.strip()
        file_label = file_path.split('/')[-2]
        folder_name = label_dict[file_label]
        file_path = os.path.join(r'C:\TBX11K\imgs', file_path)
        destination_path = os.path.join(val_destination_folder, folder_name)
        os.makedirs(destination_path, exist_ok=True)
        move_file(file_path, destination_path)

#INI DI SPLITTT FILNYAAA ELAHH BODOH
import xml.etree.ElementTree as ET

# Mendefinisikan path folder yang berisi file XML
folder_path = r"C:\TBX11K\annotations\xml"

path = {
    'atb': [],
    'ltb': [],
    'altb': []
}
num_bound = []

# loop setiap file dalam folder
for filename in os.listdir(folder_path):
    if filename.endswith(".xml"):
        file_path = os.path.join(folder_path, filename)

        # Membuka file XML
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Mengambil elemen yang berisi nama kelas
        object_elems = root.findall("object")
        num_bound.append(len(object_elems))
        classes_name = []

        for object_elem in object_elems:
            name_elem = object_elem.find("name")
            class_name = name_elem.text
            classes_name.append(class_name)

        if len(classes_name) > 1:
            n = 0
            for i in range(len(classes_name) - 1):
                if classes_name[i] != classes_name[i + 1]:
                    n += 1

            if n != 0:
                path['altb'].append(filename)
            elif classes_name[0] == 'ActiveTuberculosis':
                path['atb'].append(filename)
            else:
                path['ltb'].append(filename)
        else:
            if class_name == 'ActiveTuberculosis':
                path['atb'].append(filename)
            else:
                path['ltb'].append(filename)

import os
import shutil

# Fungsi untuk memindahkan file ke folder tujuan
def move_file(source_path, destination_path):
    shutil.move(source_path, destination_path)

# TRAIN
# Path folder sumber
source_folder = 'train\\tb'

# Path folder tujuan
destination_folder = 'train'

# Iterasi
for folder_name, file_list in path.items():
    # Membuat path folder tujuan berdasarkan nama folder
    folder_path = os.path.join(destination_folder, folder_name)
    # Membuat folder tujuan jika belum ada
    os.makedirs(folder_path, exist_ok=True)

    # Iterasi melalui daftar nama file
    for file_name in file_list:
        # Membuat path file sumber berdasarkan nama file
        name = file_name.split('.')[0] + '.png'
        source_path = os.path.join(source_folder, name)
        # Memindahkan file ke folder tujuan
        if os.path.exists(source_path):
            move_file(source_path, folder_path)

# VAL
# Path folder sumber
source_folder = 'val\\tb'

# Path folder tujuan
destination_folder = 'val'

# Iterasi
for folder_name, file_list in path.items():
    # Membuat path folder tujuan berdasarkan nama folder
    folder_path = os.path.join(destination_folder, folder_name)
    # Membuat folder tujuan
    os.makedirs(folder_path, exist_ok=True)

    # Iterasi melalui daftar nama file
    for file_name in file_list:
        # Membuat path file sumber berdasarkan nama file
        name = file_name.split('.')[0] + '.png'
        source_path = os.path.join(source_folder, name)
        # Memindahkan file ke folder tujuan
        if os.path.exists(source_path):
            move_file(source_path, folder_path)
