import os
import shutil
import random

source_dir = r"F:\IESA_dataset\IESA"   # your current dataset folder
output_dir = r"F:\IESA_dataset\IESA_Split"    # output folder

train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

classes = os.listdir(source_dir)

for cls in classes:
    class_path = os.path.join(source_dir, cls)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for folder_name, file_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        save_folder = os.path.join(output_dir, folder_name, cls)
        os.makedirs(save_folder, exist_ok=True)

        for file in file_list:
            src_file = os.path.join(class_path, file)
            dst_file = os.path.join(save_folder, file)
            shutil.copy(src_file, dst_file)

    print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

print("✅ Dataset split completed!")
print("Saved in:", output_dir)
import os
import shutil
import random
source_dir = r"F:\IESA_dataset\IESA"
output_dir = r"F:\IESA_dataset\IESA_Split"

train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

classes = os.listdir(source_dir)

for cls in classes:
    class_path = os.path.join(source_dir, cls)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for folder_name, file_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        save_folder = os.path.join(output_dir, folder_name, cls)
        os.makedirs(save_folder, exist_ok=True)

        for file in file_list:
            src_file = os.path.join(class_path, file)
            dst_file = os.path.join(save_folder, file)
            shutil.copy(src_file, dst_file)

    print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

print("✅ Dataset split completed!")
print("Saved in:", output_dir)
