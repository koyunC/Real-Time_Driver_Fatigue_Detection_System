import os
import random
import shutil

def split_class(filename):
    '''
    Return: 0 for closed, 1 for open
    '''
    return 0 if filename.split('_')[2] == '0' else 1

def collect_images(source_dir, dest_dir, total_images=10000, seed=42):
    random.seed(seed)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        os.makedirs(os.path.join(dest_dir, 'closed'))
        os.makedirs(os.path.join(dest_dir, 'open'))

    # closed:open = 1:1
    all_images = [] # absolute paths
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png'):
                all_images.append(os.path.join(root, file))
    
    random.shuffle(all_images)

    closed_count = 0
    open_count = 0
    for img_path in all_images:
        if closed_count + open_count >= total_images:
            break
        cls = split_class(os.path.basename(img_path))
        if cls == 0 and closed_count < total_images // 2:
            shutil.copy(img_path, os.path.join(dest_dir, 'closed', os.path.basename(img_path)))
            closed_count += 1
        elif cls == 1 and open_count < total_images // 2:
            shutil.copy(img_path, os.path.join(dest_dir, 'open', os.path.basename(img_path)))
            open_count += 1

if __name__ == "__main__":
    source_directory = "./mrlEyes_2018_01"
    destination_directory = "./images"
    collect_images(source_directory, destination_directory, total_images=10000)
