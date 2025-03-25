import os
import sys
import numpy as np
import pandas as pd
import yaml
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger


def read_yaml(file_path: str) -> dict:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file: {file_path} is not exists")
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error reading the yaml file: {e}")
        raise CustomException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def logistic_regression_load_object(file_path: str, ) -> LogisticRegression:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file: {file_path} is not exists")
        # Load the LogisticRegression model using pickle
        with open("final_model/logistic_regression.pkl", "rb") as file:
            model: LogisticRegression = pickle.load(file)
            return model
    except Exception as e:
        raise CustomException(e, sys) from e


def sentence_transformer_save_object(file_path: str, obj: SentenceTransformer) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        obj.save(file_path)
        logger.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def sentence_transformer_load_object(file_path: str, ) -> SentenceTransformer:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file: {file_path} is not exists")
        return SentenceTransformer(file_path)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of a file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of a file to load
    return: np.array data loaded
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_dataframe(df: pd.DataFrame, file_path: str, description: str):
    try:
        if df.empty:
            raise ValueError(f"Dataframe is empty")
        # create folder if not exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False, header=True)
        logger.info(f"{description} saved successfully to {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving {description}: {str(e)}", sys)

import shutil
import os

import shutil
import os

def copy_file_with_validation(source: str, destination: str, valid_extensions=None) -> None:
    """
    Copies a file from the source path to the destination path after validating the file type.

    Args:
        source (str): The file path of the source file to copy.
        destination (str): The file path where the file should be copied to.
        valid_extensions (list or None): A list of valid file extensions (e.g., ['.txt', '.csv']).
                                         If None, validation is skipped.

    Returns:
        None
    """
    try:
        # Check if the source file exists
        if not os.path.isfile(source):
            message: str = f"The source file does not exist: {source}"
            logger.error(message)
            raise FileNotFoundError(message)

        # Validate the file extension
        if valid_extensions:
            _, file_extension = os.path.splitext(source)
            if file_extension not in valid_extensions:
                message: str = f"Invalid file type. Allowed extensions: {', '.join(valid_extensions)}"
                logger.error(message)
                raise ValueError(message)

        # Copy the file
        shutil.copy(source, destination)
        logger.info(f"File copied successfully from {source} to {destination}")

    except PermissionError as pe:
        message: str = f"Permissions error when copying the file from {source} to {destination}: {str(pe)}"
        logger.error(message)
        raise CustomException(message, sys)
    except Exception as e:
        message: str = f"An error occurred when copying the file from {source} to {destination}: {str(e)}"
        logger.error(message)
        raise CustomException(message, sys)

