import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from src.log_classifier.constants import (sentence_transformer_model_name,
                                          dbscan_eps,
                                          dbscan_min_samples,
                                          dbscan_metric,
                                          cluster_label,
                                          regex_label)
from src.log_classifier.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.log_classifier.entity.config_entity import DataTransformationConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger
from src.log_classifier.utils.classifiers.regex_classifier import regex_classifier
from src.log_classifier.utils.utils import save_numpy_array_data, sentence_transformer_save_object, save_dataframe


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, config: DataTransformationConfig):
        try:
            self.class_name = self.__class__.__name__
            self.data_validation_artifact = data_validation_artifact
            self.config = config
        except Exception as e:
            logger.error(f"Error in initializing DataTransformation: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(f"Error reading the data: {e}", sys) from e

    def save_model(self, model: SentenceTransformer):
        tag: str = f"{self.class_name}::save_model"
        try:
            os.makedirs(self.config.data_transformation_sentence_transformer_folder, exist_ok=True)
            sentence_transformer_save_object(self.config.data_transformation_sentence_transformer_file_path, model)
            logger.info(f"{tag}::Model saved to {self.config.data_transformation_sentence_transformer_file_path}")
        except Exception as e:
            raise CustomException(f"Error saving model: {str(e)}", sys)

    def generate_embeddings(self, model: SentenceTransformer, data: pd.DataFrame) -> list:
        try:
            tag: str = f"{self.class_name}::generate_embeddings"
            return model.encode(data['log_message'].values.tolist())
        except Exception as e:
            raise CustomException(f"Error generating embeddings: {str(e)}", sys)

    def perform_clustering(self, embeddings: list) -> list:
        try:
            tag: str = f"{self.class_name}::perform_clustering"
            dbscan = DBSCAN(eps=dbscan_eps,
                            min_samples=dbscan_min_samples,
                            metric=dbscan_metric)
            logger.info(f"{tag}::Performing clustering")
            return dbscan.fit_predict(embeddings)
        except Exception as e:
            raise CustomException(f"Error performing clustering: {str(e)}", sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        tag: str = f"{self.class_name}::initiate_data_transformation"
        try:
            logger.info(f"{tag}::Initiating data transformation")
            # Validation check
            if not self.data_validation_artifact.validation_status:
                raise CustomException("Data validation failed", sys)

            # Step 1: Read data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)

            # Step 2: Load and save model
            model = SentenceTransformer(sentence_transformer_model_name)
            self.save_model(model)
            logger.info(f"{tag}::Model saved successfully")

            # Step 3: Generate and save embeddings
            embeddings = self.generate_embeddings(model, train_df)
            save_numpy_array_data(self.config.embeddings_file_path, embeddings)
            logger.info(f"{tag}::Embeddings saved successfully")

            # Step 4: Perform clustering
            train_df[cluster_label] = self.perform_clustering(embeddings)

            # Step 5: Classify using regex
            train_df[regex_label] = train_df['log_message'].apply(regex_classifier)

            # Step 6: Split data
            none_train_df = train_df[train_df[regex_label].isnull()]
            classified_train_df = train_df[train_df[regex_label].notnull()]

            # Step 7: Save data
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)
            save_dataframe(train_df, self.config.transformed_data_file_path, f"{self.class_name}::Transformed data")
            save_dataframe(none_train_df, self.config.transformed_none_regex_file_name, f"{self.class_name}::None regex data")
            save_dataframe(classified_train_df, self.config.transformed_classified_regex_file_name, f"{self.class_name}::Classified regex data")

            return DataTransformationArtifact(
                model_embeddings_file_path=self.config.embeddings_file_path,
                transformed_data_file_path=self.config.transformed_data_file_path,
                regex_none_classified_data_file_path=self.config.transformed_none_regex_file_name,
                regex_classified_data_file_path=self.config.transformed_classified_regex_file_name,
                sentence_transformer_file_path=self.config.data_transformation_sentence_transformer_file_path
            )
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)
