from fastapi import APIRouter
from keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow import expand_dims
from keras.utils import load_img, img_to_array
import tensorflow as tf
import re
import warnings
warnings.filterwarnings('ignore')

router = APIRouter()

# ------------------------------------------ CONFIGURATION -----------------------------------------#

model = load_model("models/fruits_and_vegetables_classification.h5", compile=False)
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.load_weights("models/fruits_and_vegetables_classification_weights.h5")

# getting the model input shape (224, 224)
target_size = model.input_shape[1:3]
dict_df = pd.read_csv("dataset/fruits_and_vegetables-class_dict.csv")
class_name_mapping = dict_df['class']

# FUNCTION

@router.get('/')
def extract_features(img_path, model):
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
    img = load_img(img_path, target_size=(224, 224, 3))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(expanded_img_array)
    predicted_class = class_name_mapping[np.argmax(result[0])]
    confidence = round((np.max(result[0])), 2)
    return predicted_class, confidence

# ------------------------------------------- RUN PROGRAM -------------------------------------------#

if __name__ == "__main__":
    img_path = "test_data/data-11.jpeg"
    predicted_class, confidence = extract_features(img_path, model)    
    result = re.sub(r'[!@#$_()]', ' ', predicted_class).replace("  ", "")
    print(f"predicted class : {result}",)
    print(f"prediction result confidence : {confidence * 100}%")