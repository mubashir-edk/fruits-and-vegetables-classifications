# from fastapi import APIRouter, File, UploadFile
# from io import BytesIO
# import os
# from PIL import Image
# from keras.models import load_model
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow import expand_dims
# from keras.utils import load_img, img_to_array

# router = APIRouter()

# # ------------------------------------------ CONFIGURATION -----------------------------------------#

# model = load_model("../artifacts/models/trained_model.h5", compile=False)
# model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# model.load_weights("../artifacts/models/trained_model.h5")

# # getting the model input shape (224, 224)
# target_size = model.input_shape[1:3]
# dict_df = pd.read_csv("../artifacts/dataset/fruits_and_vegetables-class_dict.csv")
# class_name_mapping = dict_df['class']

# # FUNCTION
# def extract_features(image_to_test, model):
#     """
#         Here we take the image that has been uploaded by the user.
#         1. load_img - Loads an image into PIL format with the target size(it can be changed)
#         2. img_to_array - converting the img to an array
#         3. expand_dims - Expand the shape of an array
#         4. model.predict - EffectiveNet model that has been used to predict the result
#         5. np.argmax(predictions[0]) - returns the index of the maximum value
#         6. class_name_mapping[np.argmax(predictions[0])] - returns the name of the class
#         7. round((np.max(predictions[0])), 2) - rounds the value to 2 decimal places
#         8. return predicted_class, confidence
#     """
#     img = load_img(image_to_test, target_size=(224, 224))  # Ensure target_size matches model input size
#     img_array = img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     result = model.predict(expanded_img_array)
#     predicted_class = class_name_mapping[np.argmax(result[0])]
#     confidence = round((np.max(result[0])), 2)
#     return predicted_class, confidence

# # FastAPI endpoint for image upload and prediction
# @router.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     # Check if the uploaded file is an image
#     if file.content_type.startswith('image/'):
        
#         file_location = f"images/{file.filename}"
#         # Read the image content
#         with open(file_location, "wb") as f:
#             contents = await file.read()
#             f.write(contents)
        
#         # Call your prediction function
#         predicted_class, confidence = extract_features(file_location, model)
        
#         os.remove(file_location)

#         # Return the prediction result as a JSON response
#         return f"predicted_class: {predicted_class}, confidence: {confidence}"
#     else:
#         return f"error: Uploaded file is not an image"