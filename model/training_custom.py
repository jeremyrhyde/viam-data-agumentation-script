import tensorflow as tf

def add_data_augmentation_layer():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomTranslation(0.3, 0.1),
        ]
    )

    return data_augmentation