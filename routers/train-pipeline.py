import os
from fastapi import FastAPI, APIRouter
import numpy as np
from ..functions.train import define_paths
from ..functions.train import split_data
from ..functions.train import create_generators

router = APIRouter()

data_dir = os.listdir("../artifacts/dataset/images")

@router.get("/train")
async def train_model(data_dir):
    # Step 1 split the data in to train, validation and test
    file_paths, labels = define_paths(data_dir)
    
    # Step 2 split the data in to train, validation and test
    train_data, valid_data, test_data = split_data(file_paths, labels)
    
    # Calculating batch size
    total_samples = train_data.samples 
    num_batches = len(train_data)
    batch_size = np.ceil(total_samples / num_batches)
    
    # Step 3 create generators
    train_gen, valid_gen, test_gen = create_generators(train_data, valid_data, test_data, batch_size)
    
    steps_per_epoch = 100
    epochs = 40
    patience = 5
    threshold = 0.9
    factor = 0.5
    
    # Step 4 Train the model
    trained_model = train_model(train_gen, valid_gen, test_gen, batch_size)
    
    
    
    
