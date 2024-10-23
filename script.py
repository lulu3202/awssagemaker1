# Writes the script to a file named script.py in a Jupyter Notebook


# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier  # For creating the Random Forest model
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, precision_score)  # For evaluating model performance
import sklearn  # Import the scikit-learn package to check version
import joblib  # Used to save and load models
import boto3  # AWS SDK for Python, used to interact with S3 or other AWS services (not used directly in this code)
import pathlib  # For path manipulation
from io import StringIO  # For handling in-memory text streams
import argparse  # For parsing command-line arguments
import os  # For interacting with the operating system (e.g., environment variables)
import numpy as np  # Used for handling arrays (not used in the code directly)
import pandas as pd  # For reading and manipulating tabular data


# Function for loading a model during inference time
def model_fn(model_dir):
    """
    Load the trained model from the model directory.
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))  # Load model saved in model_dir using joblib

# Main function (runs during training)
if __name__ == "__main__":

    print("[Info] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Define hyperparameters for the RandomForestClassifier
    parser.add_argument("--n_estimators", type=int, default=100)  # Number of trees in the forest
    parser.add_argument("--random_state", type=int, default=0)  # Seed for reproducibility

    # Define data, model, and output directories from the environment (these come from SageMaker)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))  # Directory where the model is saved
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))  # Training data directory
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))  # Testing data directory
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")  # Training dataset file name
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")  # Testing dataset file name

    # Parse the arguments provided during the execution of the script
    args, _ = parser.parse_known_args()

    # Print versions of important libraries
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    # Reading training and testing data
    print("[INFO] Reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))  # Load the training dataset
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))  # Load the testing dataset

    # Separate features and labels (assumes the last column is the label)
    features = list(train_df.columns)  # Get the column names of the training dataset
    label = features.pop(-1)  # Assume the last column is the label

    # Prepare the training and testing datasets
    print("Building training and testing datasets")
    X_train = train_df[features]  # Features for training
    X_test = test_df[features]  # Features for testing
    y_train = train_df[label]  # Labels for training
    y_test = test_df[label]  # Labels for testing

    # Print the features and label information for clarity
    print('Column order: ')
    print(features)
    print()
    print("Label column is: ", label)
    print()

    # Print the shapes of the training and testing data
    print("Data Shape: ")
    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("---- SHAPE OF TESTING DATA (15%) ----")
    print(X_test.shape)
    print(y_test.shape)
    print()

    # Train the RandomForest model
    print("Training RandomForest Model ....")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, 
                                   verbose=2, n_jobs=1)  # Create RandomForestClassifier with specified hyperparameters
    
    model.fit(X_train, y_train)  # Fit the model on the training data

    print()

    # Save the trained model to the specified directory using joblib
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("Model saved at " + model_path)

    # Evaluate the model on the testing dataset
    y_pred_test = model.predict(X_test)  # Predict the labels for the test data
    test_acc = accuracy_score(y_test, y_pred_test)  # Calculate accuracy
    test_rep = classification_report(y_test, y_pred_test)  # Generate classification report

    # Print the evaluation results
    print()
    print("---- METRICS RESULTS FOR TESTING DATA ----")
    print()
    print("Total Rows are: ", X_test.shape[0])
    print('[TESTING] Model Accuracy is: ', test_acc)
    print('[TESTING] Testing Report: ')
    print(test_rep)
