import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model

def build_model(img_size=224, num_classes=8):
    # Input is grayscale (1 channel)
    input_tensor = Input(shape=(img_size, img_size, 1), name="input")

    # Convert grayscale → 3 channel (needed for MobileNet pretrained weights)
    x = Concatenate()([input_tensor, input_tensor, input_tensor])

    base_model = MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # Freeze pretrained layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model
