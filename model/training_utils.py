import json
import os
import typing as ty
import tensorflow as tf
import numpy as np

single_label = "MODEL_TYPE_SINGLE_LABEL_CLASSIFICATION"
multi_label = "MODEL_TYPE_MULTI_LABEL_CLASSIFICATION"
labels_filename = "labels.txt"

def parse_filenames_and_labels_from_json(
    filename: str, all_labels: ty.List[str]
) -> ty.Tuple[ty.List[str], ty.List[str]]:
    """Load and parse JSON file to return image filenames and corresponding labels.
    Args:
        filename: JSONLines file containing filenames and labels
        model_type: either 'single_label' or 'multi_label'
    """
    image_filenames = []
    image_labels = []

    with open(filename, "rb") as f:
        for line in f:
            json_line = json.loads(line)
            image_filenames.append(json_line["image_path"])
            
            annotations = json_line["classification_annotations"]
            labels = []
            for annotation in annotations:
                if annotation["annotation_label"] in all_labels:
                    labels.append(annotation["annotation_label"])
            image_labels.append(labels)
    return image_filenames, image_labels

def save_labels(labels: ty.List[str], model_dir: str) -> None:
    filename = os.path.join(model_dir, labels_filename)
    with open(filename, "w") as f:
        for label in labels[:-1]:
            f.write(label + "\n")
        f.write(labels[-1])

def get_rounded_number(val: tf.Tensor, rounding_digits: int) -> tf.Tensor:
    if np.isnan(val) or np.isinf(val):
        return -1
    else:
        return float(round(val, rounding_digits))

def parse_image_and_encode_labels(
    filename: str,
    labels: ty.List[str],
    all_labels: ty.List[str],
    model_type: str,
    img_size: ty.Tuple[int, int] = (256, 256),
) -> ty.Tuple[tf.Tensor, tf.Tensor]:
    """Returns a tuple of normalized image array and hot encoded labels array.
    Args:
        filename: string representing path to image
        labels: list of up to N_LABELS associated with image
        all_labels: list of all N_LABELS
        model_type: string single_label or multi_label
    """
    image_decoded = check_type_and_decode_image(filename)

    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [img_size[0], img_size[1]])
    # Convert string labels to encoded labels
    labels_encoded = encoded_labels(labels, all_labels, model_type)
    return image_resized, labels_encoded


def decode_image(image):
    """Decodes the image as an uint8 dense vector
    Args:
        image: the image file contents as a tensor
    """
    return tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
        dtype=tf.dtypes.uint8,
    )

def check_type_and_decode_image(image_string_tensor):
    image_string = tf.io.read_file(image_string_tensor)
    return decode_image(image_string)

def encoded_labels(
    image_labels: ty.List[str], all_labels: ty.List[str], model_type: str
) -> tf.Tensor:
    if model_type == single_label:
        encoder = tf.keras.layers.StringLookup(
            vocabulary=all_labels, num_oov_indices=0, output_mode="one_hot"
        )
    elif model_type == multi_label:
        encoder = tf.keras.layers.StringLookup(
            vocabulary=all_labels, num_oov_indices=0, output_mode="multi_hot"
        )
    return encoder(image_labels)