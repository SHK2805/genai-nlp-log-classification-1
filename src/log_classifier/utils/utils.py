import sys
import yaml
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error reading the yaml file: {e}")
        raise CustomException(e, sys) from e