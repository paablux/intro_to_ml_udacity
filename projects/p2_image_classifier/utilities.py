from tensorflow_hub import KerasLayer
from numpy import asarray, expand_dims
import tensorflow as tf
from PIL import Image

from collections import OrderedDict
from json import load


def process_image(image):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_resized = tf.image.resize(image, [224, 224])
    image_norm = tf.cast(image_resized, tf.float32)
    image_norm /= 255
    return image_norm.numpy()

def predict(image_path, model, k):
    im = Image.open(image_path)
    test_image = asarray(im)
    processed_test_image = process_image(test_image)
    input_image_processed = expand_dims(processed_test_image, axis=0)
    prediction = model.predict(input_image_processed)
    top_k_values, top_k_indices = tf.nn.top_k(prediction, k=k)
    return top_k_values.numpy().tolist()[0], top_k_indices.numpy().tolist()[0]

def load_saved_model(model_path):
    return tf.keras.models.load_model(
        model_path, custom_objects={'KerasLayer': KerasLayer}
    )

def load_json_mapping(json_path):
    with open(json_path, 'r') as f:
        class_names = load(f)
        class_names = {int(key): value for key, value in class_names.items()}
        class_names_ordered = OrderedDict(sorted(class_names.items()))
        return  [value for key, value in class_names_ordered.items()]
