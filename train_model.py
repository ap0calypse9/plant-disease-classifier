import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


DATA_DIR = "data/New Plant Diseases Dataset(Augmented)/train"
VAL_DIR = "data/New Plant Diseases Dataset(Augmented)/valid"
IMG_SIZE = (128, 128)  
BATCH_SIZE = 16
EPOCHS = 5
SAMPLES_PER_CLASS = 200  # balanced sampling



def get_balanced_filepaths(data_dir, samples_per_class):
    filepaths = []
    class_names = sorted(os.listdir(data_dir))
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".jpg")]
        sampled = random.sample(images, min(samples_per_class, len(images)))
        filepaths.extend(sampled)
    random.shuffle(filepaths)
    return filepaths, class_names

train_files, class_names = get_balanced_filepaths(DATA_DIR, SAMPLES_PER_CLASS)
val_files, _ = get_balanced_filepaths(VAL_DIR, 50)


# LOAD DATASETS

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    labels="inferred",
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    labels="inferred",
    label_mode="categorical"
)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)



def build_model(num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = build_model(len(class_names))
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


model.save("plant_disease_model.h5")
print("Model saved as plant_disease_model.h5")


plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")   # save accuracy plot
plt.show()
print(" Saved accuracy plot as accuracy_plot.png")

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  
plt.show()
print(" Saved confusion matrix as confusion_matrix.png")



print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
