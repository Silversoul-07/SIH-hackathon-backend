import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Enable oneDNN optimizations

from keras.models import load_model  # TensorFlow is required for Keras to work
from keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

np.set_printoptions(suppress=True)

# Define a custom class for DepthwiseConv2D
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' from kwargs if it exists
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# Load the model with custom objects
model = load_model("app/model.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Load the labels
class_names = open("app/labels.txt", "r").read().splitlines()

async def predict_image_class(image: Image):
    if not isinstance(image, Image.Image):
        raise ValueError("Image must be a PIL image")
    
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score