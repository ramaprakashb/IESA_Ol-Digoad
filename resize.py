import os
import cv2

input_folder = r"F:\IESA_dataset\IESA"
output_folder = r"F:\IESA_dataset\IESA_Split_224"
size = (224, 224)

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)

            img = cv2.imread(input_path)
            if img is None:
                print("Skipping:", input_path)
                continue

            resized = cv2.resize(img, size)
            output_path = os.path.join(save_dir, file)
            cv2.imwrite(output_path, resized)

print("✅ Done resizing split dataset.")
print("Saved in:", output_folder)
