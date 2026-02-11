import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model import build_model

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
DATASET_PATH = "dataset"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")

OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "wafer_model.h5")

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data Augmentation for training
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    num_classes = train_data.num_classes
    print("Class Labels:", train_data.class_indices)

    # Save class labels automatically
    with open("labels.txt", "w") as f:
        for label in train_data.class_indices.keys():
            f.write(label + "\n")

    model = build_model(img_size=IMG_SIZE, num_classes=num_classes)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n✅ Training Started...\n")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved successfully: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
