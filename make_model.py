

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


# Optional: prevent TensorFlow from grabbing all GPU memory at once
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)





DATASET_DIR = "/media/mtgama/New Volume/Developing/beet_study/dataset2000"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)


plot_metric([history_feature, history_fine], "accuracy")
plot_metric([history_feature, history_fine], "loss")
plot_metric([history_feature, history_fine], "auc")
BATCH_SIZE = 8
SEED = 42

FEATURE_EXTRACTION_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

AUTOTUNE = tf.data.AUTOTUNE



train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode="binary"
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="binary"
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="binary"
)

class_names = train_ds.class_names

print("Class names:", class_names)
print("Train class names:", train_ds.class_names)
print("Val class names:", val_ds.class_names)
print("Test class names:", test_ds.class_names)




def count_images_per_class(directory):
    print(f"\nCounting images in: {directory}")

    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)

        if os.path.isdir(class_path):
            files = [
                file for file in os.listdir(class_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            ]
            print(f"{class_name}: {len(files)}")


count_images_per_class(TRAIN_DIR)
count_images_per_class(VAL_DIR)
count_images_per_class(TEST_DIR)


plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(min(9, images.shape[0])):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        label_index = int(labels[i].numpy()[0])
        plt.title(class_names[label_index])

        plt.axis("off")

plt.show()



train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)




data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation"
)




preprocess_input = keras.applications.resnet50.preprocess_input

base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)

base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,), name="input_image")

x = data_augmentation(inputs)
x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
x = layers.Dropout(0.4, name="dropout")(x)

outputs = layers.Dense(1, activation="sigmoid", name="binary_classifier")(x)

model = keras.Model(inputs, outputs, name="beet_resnet50_transfer_learning")


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]
)

model.summary()




callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="best_resnet50_beet_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]




history_feature = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FEATURE_EXTRACTION_EPOCHS,
    callbacks=callbacks
)




base_model.trainable = True

# Freeze most layers, fine-tune only the last layers
fine_tune_at = len(base_model.layers) - 20

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Important: keep BatchNormalization layers frozen during fine-tuning
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]
)

model.summary()


history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)




test_results = model.evaluate(test_ds, verbose=1)

print("\nTest results:")
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")



y_true = []
y_prob = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)

    y_prob.extend(probs.ravel())
    y_true.extend(labels.numpy().ravel())

y_true = np.array(y_true).astype(int)
y_prob = np.array(y_prob)

y_pred = (y_prob >= 0.5).astype(int)


print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

def plot_metric(history_list, metric_name):
    train_values = []
    val_values = []

    for history in history_list:
        train_values.extend(history.history.get(metric_name, []))
        val_values.extend(history.history.get(f"val_{metric_name}", []))

    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label=f"Train {metric_name}")
    plt.plot(val_values, label=f"Validation {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Training and Validation {metric_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_metric([history_feature, history_fine], "accuracy")
plot_metric([history_feature, history_fine], "loss")
plot_metric([history_feature, history_fine], "auc")
