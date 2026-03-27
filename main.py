import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

import os
print(os.listdir("backend"))
# =====================
# SETTINGS
# =====================
IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_PATH = "backend/archive/train"
TEST_PATH = "backend/archive/test"
# =====================
# LOAD DATASET
# =====================
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# =====================
# NORMALIZATION
# =====================
normalization_layer = layers.Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# =====================
# DATA AUGMENTATION
# =====================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =====================
# LOAD MODEL
# =====================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # freeze initially

# =====================
# BUILD MODEL
# =====================
x = data_augmentation(base_model.input)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# 7 emotions (you have 7 folders)
output = layers.Dense(7, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# =====================
# COMPILE
# =====================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# TRAIN
# =====================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =====================
# FINE-TUNING (IMPORTANT)
# =====================
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# =====================
# EVALUATE
# =====================
test_loss, test_acc = model.evaluate(test_data)
print("Test Accuracy:", test_acc)

# =====================
# SAVE MODEL
# =====================
model.save("emotion_model.h5")

print("Model saved successfully!")