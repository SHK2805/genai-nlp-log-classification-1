import os
from src.log_classifier.config.configuration import TrainingPipelineConfig
from src.log_classifier.constants import (data_file_folder_name,
                                          data_file_name,
                                          data_ingestion_dir_name,
                                          data_ingestion_feature_store_dir_name,
                                          data_ingestion_ingested_data_dir_name,
                                          train_file_name,
                                          data_validation_dir_name,
                                          data_validation_valid_dir)


class DataIngestionConfig:
    def __init__(self, config: TrainingPipelineConfig):
        self.class_name = self.__class__.__name__
        self.data_source_path = os.path.join(data_file_folder_name,
                                               data_file_name)
        self.data_ingestion_dir = os.path.join(config.artifact_dir,
                                               data_ingestion_dir_name)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir,
                                                    data_ingestion_feature_store_dir_name,
                                                    data_file_name)
        self.training_file_path = os.path.join(self.data_ingestion_dir,
                                               data_ingestion_ingested_data_dir_name,
                                               train_file_name)

        # folder structure
        # artifacts
        #   - data_ingestion
        #       - feature_store
        #           - synthetic_logs.csv
        #       - ingested
        #           - train_data.csv

class DataValidationConfig:
    def __init__(self, config: TrainingPipelineConfig):
        self.class_name = self.__class__.__name__
        # folders
        self.data_validation_dir = os.path.join(config.artifact_dir,
                                                data_validation_dir_name)
        self.valid_data_dir = os.path.join(self.data_validation_dir, data_validation_valid_dir)
        # files
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, train_file_name)
        # folder structure
        # - artifacts
        #   - data_validation
        #       - validated
        #           - train_data.csv