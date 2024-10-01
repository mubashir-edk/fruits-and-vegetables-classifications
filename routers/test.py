from fastapi import APIRouter
from io import BytesIO
import os
from PIL import Image
from keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow import expand_dims
from keras.utils import load_img, img_to_array
import io
import requests
from models.image_model import ImageData

router = APIRouter()

# ------------------------------------------ CONFIGURATION -----------------------------------------#

# FUNCTION
def extract_features(image_to_test, model, class_name_mapping):
    """
        Here we take the image that has been uploaded by the user.
        1. load_img - Loads an image into PIL format with the target size(it can be changed)
        2. img_to_array - converting the img to an array
        3. expand_dims - Expand the shape of an array
        4. model.predict - EffectiveNet model that has been used to predict the result
        5. np.argmax(predictions[0]) - returns the index of the maximum value
        6. class_name_mapping[np.argmax(predictions[0])] - returns the name of the class
        7. round((np.max(predictions[0])), 2) - rounds the value to 2 decimal places
        8. return predicted_class, confidence
    """
    img = load_img(image_to_test, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(expanded_img_array)
    predicted_class = class_name_mapping[np.argmax(result[0])]
    confidence = round((np.max(result[0])), 2)
    return predicted_class, confidence

# FastAPI endpoint for image upload and prediction
@router.post("/predict")
async def predict_image(image_data: ImageData):
    
    model = load_model("trained_model/fruits_and_vegetables_classification.h5", compile=False)
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.load_weights("trained_model/fruits_and_vegetables_classification.weights.h5")

    target_size = model.input_shape[1:3]
    dict_df = pd.read_csv("artifacts/dataset/fruits_and_vegetables-class_dict.csv") 
    class_name_mapping = dict_df['class'].tolist()
    
    try:
        # Fetch the image from the URL
        response = requests.get(image_data.image_url)

        # Open the image with PIL
        img_bytes = BytesIO(response.content)
        image = Image.open(img_bytes)

        # Resize the image
        target_size = (224, 224)
        image = image.resize(target_size)

        # Save the image temporarily to pass it to load_img
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # predict
        predicted_class, confidence = extract_features(temp_image_path, model, class_name_mapping)

        # remove the temporary image
        os.remove(temp_image_path)

        # Ensure confidence is formatted correctly
        if isinstance(confidence, np.float32):
            confidence = float(confidence)  # Convert to standard float
            
        confidence_threshold = 0.9
        
        if predicted_class not in class_name_mapping or confidence < confidence_threshold:
            return {
                "predicted_class": "None",
                "confidence": confidence
            }

        # Return the prediction result
        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}