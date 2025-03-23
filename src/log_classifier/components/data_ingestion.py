import os
import sys
import pandas as pd
from src.log_classifier.entity.config_entity import DataIngestionConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.class_name = self.__class__.__name__
            self.data_ingestion_config = data_ingestion_config
            self.data_file_path = self.data_ingestion_config.data_file_path
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)

    def is_datafile_exists(self):
        try:
            if not self.data_file_path:
                raise CustomException(f"Data file path is empty", sys)
            if not os.path.exists(self.data_file_path):
                raise CustomException(f"Data file does not exist at {self.data_file_path}", sys)

            return True
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)

    def read_data_as_dataframe(self):
        try:
            if not self.is_datafile_exists():
                raise CustomException(f"Data file does not exist at {self.data_file_path}", sys)
            df = pd.read_csv(self.data_file_path)
            return df
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)