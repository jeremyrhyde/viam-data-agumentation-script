import argparse
import json
import os
import typing as ty
import tensorflow as tf
from keras import Model, callbacks
import numpy as np
from model import *

single_label = "MODEL_TYPE_SINGLE_LABEL_CLASSIFICATION"
multi_label = "MODEL_TYPE_MULTI_LABEL_CLASSIFICATION"
metrics_filename = "model_metrics.json"
labels_filename = "labels.txt"

TFLITE_OPS = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]

TFLITE_OPTIMIZATIONS = [tf.lite.Optimize.DEFAULT]

ROUNDING_DIGITS = 5

# Normalization parameters are required when reprocessing the image.
_INPUT_NORM_MEAN = 127.5
_INPUT_NORM_STD = 127.5

def get_neural_network_params(
    num_classes: int, model_type: str
) -> ty.Tuple[str, str, str, str]:
    """Function that returns units and activation used for the last layer
        and loss function for the model, based on number of classes and model type.
    Args:
        labels: list of labels corresponding to images
        model_type: string single-label or multi-label for desired output
    """
    # Single-label Classification
    if model_type == single_label:
        units = num_classes
        activation = "softmax"
        loss = tf.keras.losses.categorical_crossentropy
        metrics = (
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        )
    # Multi-label Classification
    elif model_type == multi_label:
        units = num_classes
        activation = "sigmoid"
        loss = tf.keras.losses.binary_crossentropy
        metrics = (
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        )
    return units, activation, loss, metrics

def preprocessing_layers_classification(
    img_size: ty.Tuple[int, int] = (256, 256)
) -> ty.Tuple[tf.Tensor, tf.Tensor]:
    """Preprocessing steps to apply to all images passed through the model.
    Args:
        img_size: optional 2D shape of image
    """
    preprocessing = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(
                img_size[0], img_size[1], crop_to_aspect_ratio=False
            ),
        ]
    )
    return preprocessing

def create_dataset_classification(
    filenames: ty.List[str],
    labels: ty.List[str],
    all_labels: ty.List[str],
    model_type: str,
    img_size: ty.Tuple[int, int] = (256, 256),
    train_split: float = 0.8,
    batch_size: int = 64,
    shuffle_buffer_size: int = 1024,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    prefetch_buffer_size: int = tf.data.experimental.AUTOTUNE,
) -> ty.Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load and parse dataset from Tensorflow datasets.
    Args:
        filenames: string list of image paths
        labels: list of string lists, where each string list contains up to N_LABEL labels associated with an image
        all_labels: string list of all N_LABELS
        model_type: string single_label or multi_label
    """
    # Create a first dataset of file paths and labels
    if model_type == single_label:
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (filenames, tf.ragged.constant(labels))
        )

    def mapping_fnc(x, y):
        return parse_image_and_encode_labels(x, y, all_labels, model_type, img_size)

    # Parse and preprocess observations in parallel
    dataset = dataset.map(mapping_fnc, num_parallel_calls=num_parallel_calls)

    # Shuffle the data for each buffer size
    # Disabling reshuffling ensures items from the training and test set will not get shuffled into each other
    dataset = dataset.shuffle(
        buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False
    )

    train_size = int(train_split * len(filenames))

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    # Batch the data for multiple steps
    # If the size of training data is smaller than the batch size,
    # batch the data to expand the dimensions by a length 1 axis.
    # This will ensure that the training data is valid model input
    train_batch_size = batch_size if batch_size < train_size else train_size
    if model_type == single_label:
        train_dataset = train_dataset.batch(train_batch_size)
    else:
        train_dataset = train_dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(train_batch_size)
        )

    # Fetch batches in the background while the model is training.
    train_dataset = train_dataset.prefetch(buffer_size=prefetch_buffer_size)

    return train_dataset, test_dataset


# Build the Keras model
def build_and_compile_classification(
    labels: ty.List[str], model_type: str, input_shape: ty.Tuple[int, int, int]
) -> Model:
    units, activation, loss_fnc, metrics = get_neural_network_params(
        len(labels), model_type
    )

    x = tf.keras.Input(input_shape, dtype=tf.uint8)
    # Data processing
    preprocessing = preprocessing_layers_classification(input_shape[:-1])

    # Get the pre-trained model
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    # Add custom layers
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()
    # Output layer
    classification = tf.keras.layers.Dense(units, activation=activation, name="output")

    y = tf.keras.Sequential(
        [
            preprocessing,
            add_data_augmentation_layer(),   # custom data augmentation add-on
            base_model,
            global_pooling,
            classification,
        ]
    )(x)

    model = tf.keras.Model(x, y)

    model.compile(
        loss=loss_fnc,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[metrics],
    )
    return model

def save_model_metrics_classification(
    loss_history: callbacks.History,
    monitored_val: ty.List[str],
    model_dir: str,
    model: Model,
    test_dataset: tf.data.Dataset,
) -> None:
    test_images = np.array([x for x, _ in test_dataset])
    test_labels = np.array([y for _, y in test_dataset])

    test_metrics = model.evaluate(test_images, test_labels)

    metrics = {}
    # Since there could be potentially many occurences of the maximum value being monitored,
    # we reverse the list storing the tracked values and take the last occurence.
    monitored_metric_max_idx = len(monitored_val) - np.argmax(monitored_val[::-1]) - 1
    for i, key in enumerate(model.metrics_names):
        metrics["train_" + key] = get_rounded_number(
            loss_history.history[key][monitored_metric_max_idx], ROUNDING_DIGITS
        )
        metrics["test_" + key] = get_rounded_number(test_metrics[i], ROUNDING_DIGITS)

    # Save the loss and test metrics as model metrics
    filename = os.path.join(model_dir, metrics_filename)
    with open(filename, "w") as f:
        json.dump(metrics, f, ensure_ascii=False)

def save_tflite_classification(
    model: Model,
    model_dir: str,
    model_name: str,
    target_shape: ty.Tuple[int, int, int],
) -> None:
    # Convert the model to tflite, with batch size 1 so the graph does not have dynamic-sized tensors.
    input = tf.keras.Input(target_shape, batch_size=1, dtype=tf.uint8)
    output = model(input, training=False)
    wrapped_model = tf.keras.Model(inputs=input, outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(wrapped_model)
    converter.target_spec.supported_ops = TFLITE_OPS
    tflite_model = converter.convert()

    filename = os.path.join(model_dir, f"{model_name}.tflite")
    with open(filename, "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", dest="data_json", type=str)                     # required
    parser.add_argument("--model_output_directory", dest="model_dir", type=str)           # required
    args = parser.parse_args()
    MODEL_DIR = args.model_dir
    DATA_JSON = args.data_json

    # Set up compute device strategy
    if len(tf.config.list_physical_devices("GPU")) > 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    IMG_SIZE = (256, 256)
    EPOCHS = 100
    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 512
    AUTOTUNE = (
        tf.data.experimental.AUTOTUNE
    )  # Adapt preprocessing and prefetching dynamically

    # Model constants
    NUM_WORKERS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS

    task_type = single_label

    # Read dataset file
    LABELS = ["red", "no_red"]                                           
    image_filenames, image_labels = parse_filenames_and_labels_from_json(DATA_JSON, LABELS)
    # Generate 80/20 split for train and test data
    train_dataset, test_dataset = create_dataset_classification(
        filenames=image_filenames,
        labels=image_labels,
        all_labels=LABELS,
        model_type=task_type,
        img_size=IMG_SIZE,
        train_split=0.8,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        num_parallel_calls=AUTOTUNE,
        prefetch_buffer_size=AUTOTUNE,
    )

    # Build and compile model
    with strategy.scope():
        model = build_and_compile_classification(
            LABELS, task_type, IMG_SIZE + (3,)
        )

    # Get callbacks for training classification
    callbackEarlyStopping = tf.keras.callbacks.EarlyStopping(
    # Stop training when `monitor` value is no longer improving
    monitor="categorical_accuracy",
        # "no longer improving" being defined as "no better than 'min_delta' less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 'patience' epochs"
        patience=5,
        # Restore weights from the best performing model, requires keeping track of model weights and performance.
        restore_best_weights=True,
    )
    callbackReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        # Reduce learning rate when `loss` is no longer improving
        monitor="loss",
        # "no longer improving" being defined as "no better than 'min_delta' less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 'patience' epochs"
        patience=5,
        # Default lower bound on learning rate
        min_lr=0,
    )

    # Train model on data
    loss_history = model.fit(
            x=train_dataset, epochs=EPOCHS, callbacks=[callbackEarlyStopping, callbackReduceLROnPlateau]
    )
    # Get the values of what is being monitored in the early stopping policy,
    # since this is what is used to restore best weights for the resulting model.
    monitored_val = callbackEarlyStopping.get_monitor_value(
        loss_history.history
    )
    # Save trained model metrics to JSON file
    save_model_metrics_classification(
        loss_history,
        monitored_val,
        MODEL_DIR,
        model,
        test_dataset,
    )
    # Save labels.txt file
    save_labels(LABELS, MODEL_DIR)
    # Convert the model to tflite
    save_tflite_classification(
        model, MODEL_DIR, "beepboop", IMG_SIZE + (3,)
    )
