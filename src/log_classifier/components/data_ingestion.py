import os
import sys

import pandas as pd

from src.log_classifier.entity.artifact_entity import DataIngestionArtifact
from src.log_classifier.entity.config_entity import DataIngestionConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger

# In the future, if the datasource is changed to a different source, we can modify the class
# Added for future extensibility
class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.class_name = self.__class__.__name__
        # the data is already stored in the data folder
        # we don't need to download it or configure it
        self.config = data_ingestion_config

    def export_collection_as_dataframe(self):
        """Export the collection as a dataframe."""
        tag = f"{self.class_name}::export_collection_as_dataframe"
        try:
            # check if a source file exists
            if not os.path.exists(self.config.data_source_path):
                raise FileNotFoundError(f"File not found: {self.config.data_source_path}")
            # read the data from the data source
            df = pd.read_csv(self.config.data_source_path)
            return df
        except Exception as e:
            logger.error(f"{tag}::Error in exporting collection as dataframe: {e}")
            raise CustomException(e, sys)

    def export_data_into_feature_store(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Export the data into the feature store.
        This will save the copy of the data in the feature store for versioning.
        """
        tag = f"{self.class_name}::export_data_into_feature_store"
        try:
            feature_store_file_path = self.config.feature_store_file_path
            # create the dir if not exists
            feature_store_dir = os.path.dirname(feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            logger.info(f"{tag}::Created folder: {feature_store_dir}")
            # save the data into the feature store
            df.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(f"{tag}::Exported data into feature store: {feature_store_file_path}")
            return df
        except Exception as e:
            logger.error(f"{tag}::Error in exporting data into feature store: {e}")
            raise CustomException(e, sys)

    def export_data_into_train_test(self, df: pd.DataFrame):
        """
        Export the data into the train and test files.
        Here we are only using the train data for training the model.
        We are not having any test data.
        This is added to future expansion if we want a train test split
        """
        tag = f"{self.class_name}::export_data_into_train_test"
        try:
            training_file_path = self.config.training_file_path
            # create the dir if not exists
            train_test_dir = os.path.dirname(training_file_path)
            os.makedirs(train_test_dir, exist_ok=True)
            logger.info(f"{tag}::Created folder: {train_test_dir}")
            # save the train and test data
            df.to_csv(training_file_path, index=False, header=True)
            logger.info(
                f"{tag}::Exported data into train file: {training_file_path}")
            return training_file_path
        except Exception as e:
            logger.error(f"{tag}::Error in exporting data into train and test files: {e}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        tag = f"{self.class_name}::initiate_data_ingestion"
        try:
            logger.info(f"{tag}::Initiated data ingestion")
            # read the collection from MongoDB as a dataframe
            dataframe = self.export_collection_as_dataframe()
            logger.info(f"{tag}::Completed exporting collection as dataframe")
            df = self.export_data_into_feature_store(dataframe)
            logger.info(f"{tag}::Completed exporting data into feature store")

            training_file_path = self.export_data_into_train_test(df)
            logger.info(f"{tag}::Completed exporting data into train and test files")
            logger.info(f"{tag}::Completed data ingestion")
            return DataIngestionArtifact(train_file_path=training_file_path)
        except Exception as e:
            logger.error(f"{tag}::Error in initiating data ingestion: {e}")
            raise CustomException(e, sys)
