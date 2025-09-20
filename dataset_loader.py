# dataset_loader.py
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def load_datasets(base_dir="data", img_size=(224, 224), batch_size=32, use_cache=True):
    """
    Load PlantVillage dataset from the given folder structure.
    Expects:
    data/
    New Plant Diseases Dataset(Augmented)/
        train/
            class1/
            class2/
            ...
        valid/
            class1/
            class2/
            ...
        test/
            class1/
            class2/
            ...
    Returns:
    train_ds, val_ds, test_ds, class_names
    """
    train_dir = os.path.join(base_dir, "New Plant Diseases Dataset(Augmented)", "train")
    val_dir = os.path.join(base_dir, "New Plant Diseases Dataset(Augmented)", "valid")
    test_dir = os.path.join(base_dir, "test")

    # Load datasets (force RGB)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb"
    )
    class_names = train_ds.class_names  # Save class names before transformations

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb"
    )

    # Ensure all images are RGB (convert grayscale if any)
    def ensure_rgb(image, label):
        if image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        return image, label

    train_ds = train_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE)

    # Normalize images to [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Cache on disk to avoid memory issues + prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    if use_cache:
        train_ds = train_ds.cache("train_cache.tf-data").shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache("val_cache.tf-data").prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache("test_cache.tf-data").prefetch(buffer_size=AUTOTUNE)
    else:
        train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def visualize_samples(dataset, class_names, num_images=9):
    """
    Show a grid of sample images from the dataset.
    Handles cases where batch size is smaller than num_images.
    """
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        n = min(num_images, images.shape[0])
        for i in range(n):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Use smaller batch size if memory is limited
    train_ds, val_ds, test_ds, class_names = load_datasets(batch_size=16, use_cache=False)
    print("Classes:", class_names)
    print("Train batches:", len(train_ds))
    print("Validation batches:", len(val_ds))
    print("Test batches:", len(test_ds))
    
    # Visualize some samples
    visualize_samples(train_ds, class_names)