import os
from fastapi import FastAPI, APIRouter
import numpy as np
from functions.train import define_paths
from functions.train import define_df
from functions.train import split_data
from functions.train import create_generators
from functions.train import training_model
from functions.train import save_model

router = APIRouter()

@router.get("/train")
async def train_model():
    
    data_dir = os.path.join('artifacts','dataset', 'images')
    
    # Step 1 Define the paths
    file_paths, labels = define_paths(data_dir)
    
    # Step 2 Convert to dataframe
    data_frame = define_df(file_paths, labels)
    
    # Step 3 Split the data(dataframe) in to train, validation and test
    train_df, valid_df, test_df = split_data(data_frame)
    
    # Calculating batch size
    # total_samples = train_df.samples
    # num_batches = len(train_df)
    # print(total_samples, num_batches)
    # batch_size = np.ceil(total_samples / num_batches)
    batch_size = 32
    
    # Step 4 create generators
    train_gen, valid_gen, test_gen = create_generators(train_df, valid_df, test_df, batch_size)
    
    steps_per_epoch = 20
    epochs = 100
    patience = 3
    stop_patience = 5
    threshold = 0.9
    factor = 0.5
    ask_epoch = 4
    
    # Step 5 Train the model and after that test and evaluate
    model, train_score, valid_score, test_score = training_model(test_df, train_gen, valid_gen, test_gen, batch_size, steps_per_epoch, epochs, patience, stop_patience, threshold, factor, ask_epoch)
    
    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])
    print('-' * 20)
    print("Validation Loss: ", valid_score[0])
    print("Validation Accuracy: ", valid_score[1])
    print('-' * 20)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])
    
    # Step 6 Save the trained model and its model weights
    save_model(model)
    
    return train_score, valid_score, test_score
    
