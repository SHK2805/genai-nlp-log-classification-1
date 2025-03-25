import os
import sys
import numpy as np
import pandas as pd
import yaml
import pickle

from sentence_transformers import SentenceTransformer

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