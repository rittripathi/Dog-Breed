import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow_hub as hub

@tf.keras.utils.register_keras_serializable()
class HubKerasLayerWrapper(tf.keras.layers.Layer):
    def __init__(self, function, arguments=None, **kwargs):
        super().__init__(**kwargs)
        self._function = function
        self._arguments = arguments if arguments is not None else {}
        self.hub_layer = hub.KerasLayer(self._function, trainable=True, arguments=self._arguments)

    def call(self, inputs):
        return self.hub_layer(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'function': self._function,
            'arguments': self._arguments,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract the correct model URL from the nested dictionary structure
        function_config = config.get('function')
        if isinstance(function_config, dict) and 'config' in function_config and 'closure' in function_config['config']:
            function_url = function_config['config']['closure'][0]
            config['function'] = function_url  # Update function in config with the URL
        config.pop('model_url', None)
        return cls(**config)


# Load labels
labels = pd.read_csv("/content/drive/MyDrive/DogBreed/labels.csv")

# Load model
model = tf.keras.models.load_model(
    "/content/drive/MyDrive/DogBreed/dog_breed_model.keras",
    custom_objects={"HubKerasLayerWrapper": HubKerasLayerWrapper},
    compile=False
)

st.title("🐶 Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload an image of a dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224,224))  # Resize for model
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    breed = labels.iloc[predicted_class, 0]

    st.success(f"Predicted Breed: **{breed}** 🐾")
