import os
from src.log_classifier.config.configuration import TrainingPipelineConfig
from src.log_classifier.constants import (data_file_folder_name,
                                          data_file_name,
                                          data_ingestion_dir_name,
                                          data_ingestion_feature_store_dir_name,
                                          data_ingestion_ingested_data_dir_name,
                                          train_file_name,
                                          data_validation_dir_name,
                                          data_validation_valid_dir, data_transformation_dir_name,
                                          data_transformation_embeddings_file_name,
                                          dbscan_eps, dbscan_min_samples, dbscan_metric,
                                          data_transformation_embeddings_dir, data_transformation_data_dir,
                                          data_transformation_regex_none_classified,
                                          data_transformation_regex_classified, model_trainer_dir_name)
global_data_file_name = data_file_name
global_train_data_file_name = train_file_name
class DataIngestionConfig:
    def __init__(self, config: TrainingPipelineConfig):
        self.class_name = self.__class__.__name__
        self.data_source_path = os.path.join(data_file_folder_name,
                                               global_data_file_name)
        self.data_ingestion_dir = os.path.join(config.artifact_dir,
                                               data_ingestion_dir_name)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir,
                                                    data_ingestion_feature_store_dir_name,
                                                    global_data_file_name)
        self.training_file_path = os.path.join(self.data_ingestion_dir,
                                               data_ingestion_ingested_data_dir_name,
                                               global_train_data_file_name)

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
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, global_train_data_file_name)
        # folder structure
        # - artifacts
        #   - data_validation
        #       - validated
        #           - train_data.csv

class DataTransformationConfig:
    def __init__(self, config: TrainingPipelineConfig):
        self.class_name = self.__class__.__name__
        self.data_transformation_dir = os.path.join(config.artifact_dir, data_transformation_dir_name)
        self.saved_embeddings_dir = os.path.join(self.data_transformation_dir, data_transformation_embeddings_dir)
        self.embeddings_file_path = os.path.join(self.saved_embeddings_dir, data_transformation_embeddings_file_name)
        self.transformed_data_dir = os.path.join(self.data_transformation_dir, data_transformation_data_dir)
        self.transformed_data_file_path = os.path.join(self.transformed_data_dir, global_train_data_file_name)
        self.transformed_none_regex_file_name: str = os.path.join(self.transformed_data_dir, data_transformation_regex_none_classified)
        self.transformed_classified_regex_file_name: str = os.path.join(self.transformed_data_dir, data_transformation_regex_classified)
        # dbscan clustering
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_metric = dbscan_metric

        # folder structure
        # - artifacts
        #   - data_transformation
        #       - embeddings
        #           - embeddings.npy
        #       - transformed
        #           - classified_none_train_data.csv
        #           - classified_train_data.csv
        #           - train_data.csv

class ModelTrainerConfig:
    def __init__(self, config: TrainingPipelineConfig):
        self.class_name = self.__class__.__name__
        self.model_trainer_dir = os.path.join(config.artifact_dir, model_trainer_dir_name)