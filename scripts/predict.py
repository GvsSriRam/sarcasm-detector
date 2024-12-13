import os
import tempfile

import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel


# Predictor class that holds the prediction functionality to support the trained tensorflow classification model
class Predictor:

    def __init__(self):
        # Load pre-trained image model (e.g., VGG-16)
        self.image_model = tf.keras.applications.VGG16(
            # weights='imagenet',
            weights="Notebooks/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
            include_top=False, pooling='avg'
        )

        # Load pre-trained text model (e.g., Sentence Transformer)
        text_model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)

        # Load pre-trained classification model
        model_location = "Notebooks/model.keras"
        self.model = tf.keras.models.load_model(model_location)

        # Load classes in the trained dataset
        self.classes = np.load("Data/preprocessed/classes.npy")

    def preprocess_image_for_ocr(self, image_path):
        """Preprocess the image to improve OCR accuracy."""
        # Load image
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply thresholding to binarize the image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def detect_ocr(self, image_path):
        """Extract text from a meme image."""
        # Preprocess the image
        processed_image = self.preprocess_image_for_ocr(image_path)

        # OCR: Convert image to text
        text = pytesseract.image_to_string(processed_image, lang='eng')

        return text

    def get_image_embedding(self, image_path):
        # Load and preprocess the image
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)

        image = tf.keras.applications.vgg16.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)

        # Get the feature vector
        return self.image_model.predict(image)

    def get_text_embedding(self, text):
        inputs = self.text_tokenizer(text, return_tensors='pt')
        outputs = self.text_model(**inputs)
        return outputs.pooler_output.detach().numpy()

    def predict(self, uploaded_file):
        # Step-1: Create a temp file for this uploaded file
        file_extension = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Feature extraction
        text = self.detect_ocr(temp_file_path) # OCR Detection
        image_embedding = self.get_image_embedding(temp_file_path) # Image Embeddings
        text_embedding = self.get_text_embedding(text) # Text embeddings
        combined_embedding = np.concatenate([image_embedding, text_embedding], axis=1) # Full feature vector

        # Raw prediction probabilities
        prediction = self.model.predict(combined_embedding, verbose=0)
        # print(prediction)

        # Return appropriate class label
        return self.classes[np.argmax(prediction)]
