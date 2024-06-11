import tensorflow as tf

def add_data_augmentation_layer():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    return data_augmentation