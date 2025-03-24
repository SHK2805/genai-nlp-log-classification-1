import os
import sys

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

from src.log_classifier.constants import (sentence_transformer_model_name,
                                          dbscan_eps,
                                          dbscan_min_samples,
                                          dbscan_metric,
                                          cluster_label, regex_label)
from src.log_classifier.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.log_classifier.entity.config_entity import DataTransformationConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger
from src.log_classifier.utils.classify_regex import classify_with_regex
from src.log_classifier.utils.utils import save_numpy_array_data


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 config: DataTransformationConfig):
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
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading the data: {e}")
            raise CustomException(e, sys) from e

    def visualize_data(self, data: pd.DataFrame):
        tag: str = f"{self.class_name}::visualize_data"
        try:
            # check if data is not empty
            if data.empty:
                raise CustomException("Data is empty", sys)

            cluster_counts = data[cluster_label].value_counts()
            logger.info(f"{tag}::Number of clusters: {cluster_counts}")

            for cluster in cluster_counts[cluster_counts > 10].index:
                cluster_data = data[data[cluster_label] == cluster]
                print(f"{tag}::Cluster: {cluster}")
                print(f"{tag}::Number of logs: {cluster_data.shape[0]}")
                print(f"{tag}::Sample logs\n: {cluster_data['log_message'].sample(5).values}")
                print("*" * 50)

        except Exception as e:
            message = f"{tag}::Error in visualizing data: {str(e)}"
            logger.error(message)
            raise CustomException(message, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        tag: str = f"{self.class_name}::initiate_data_transformation"
        try:
            logger.info(f"{tag}::Initiating data transformation")
            # check if data validation is successful
            if not self.data_validation_artifact.validation_status:
                raise CustomException("Data validation failed", sys)

            # read the data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            logger.info(f"{tag}::Data read successfully")

            # load the sentence transformer model
            model = SentenceTransformer(sentence_transformer_model_name)
            logger.info(f"{tag}::Model {sentence_transformer_model_name} loaded successfully")

            # get the embeddings for the log messages
            # Assuming x_feature_names contains the names of the columns to encode
            embeddings = model.encode(train_df['log_message'].values.tolist())
            logger.info(f"{tag}::Embeddings generated successfully")

            # save the embeddings as a numpy array
            # NOTE: Here we are saving only the columns in x_feature_names as a numpy array
            # save data
            data_transformation_dir = self.config.data_transformation_dir
            # create directory if not exists
            os.makedirs(data_transformation_dir, exist_ok=True)
            save_numpy_array_data(self.config.embeddings_file_path, embeddings)

            # perform the dbscan clustering
            # NOTE: see the data in the cluster and tune the dbscan parameters
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)
            clusters = dbscan.fit_predict(embeddings)

            # add the cluster labels to the dataframe
            train_df[cluster_label] = clusters

            """
            visualize the data to identify patterns
            This is done to identify if the clusters are meaningful and fine tune the dbscan parameters
            Generally this will be done in a colab notebook or a jupyter notebook to identify the patterns and prepare the regex patterns
            """
            # self.visualize_data(train_df)

            # apply regex to the dataframe and create a new column with the regex pattern
            # this will be used to identify the log message patterns
            # add the regex labels to the dataframe
            train_df[regex_label] = train_df['log_message'].apply(classify_with_regex)

            # check the data that are None
            none_train_df = train_df[train_df[regex_label].isnull()].copy()
            logger.info(f"{tag}::Number of logs with regex classification as None: {none_train_df.shape[0]}")

            # non-null data
            classified_train_df = train_df[train_df[regex_label].notnull()].copy()
            logger.info(f"{tag}::Number of logs with regex classification: {classified_train_df.shape[0]}")


            # save the dataframe with cluster labels,
            # save data
            transformed_data_dir = self.config.transformed_data_dir
            # create directory if not exists
            os.makedirs(transformed_data_dir, exist_ok=True)
            train_df.to_csv(self.config.transformed_data_file_path, index=False, header=True)
            logger.info(f"{tag}::Regex data saved successfully to {self.config.transformed_data_file_path}")

            # save the data that are not classified
            none_train_df.to_csv(self.config.transformed_none_regex_file_name, index=False, header=True)
            logger.info(f"{tag}::none Regex data saved successfully to {self.config.transformed_none_regex_file_name}")

            # save the data that are classified
            classified_train_df.to_csv(self.config.transformed_classified_regex_file_name, index=False, header=True)
            logger.info(f"{tag}::Regex classified data saved successfully to {self.config.transformed_classified_regex_file_name}")

            return DataTransformationArtifact(model_embeddings_file_path=self.config.embeddings_file_path,
                                              transformed_data_file_path=self.config.transformed_data_file_path,
                                              regex_none_classified_data_file_path=self.config.transformed_none_regex_file_name,
                                              regex_classified_data_file_path=self.config.transformed_classified_regex_file_name)
        except Exception as e:
            logger.error(f"{tag}::Error in loading data: {str(e)}")
            raise CustomException(e, sys)