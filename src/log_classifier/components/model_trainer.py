import os.path
import sys

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.log_classifier.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.log_classifier.entity.config_entity import ModelTrainerConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger
from src.log_classifier.utils.utils import sentence_transformer_load_object, save_object


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.class_name = self.__class__.__name__
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            message = f"Error in initializing ModelTrainer: {str(e)}"
            logger.error(message)
            raise CustomException(message, sys)

    def perform_bret_classification(self, non_legacy_crm_df: pd.DataFrame):
        sentence_transformer_file_path = self.data_transformation_artifact.sentence_transformer_file_path
        if not os.path.exists(sentence_transformer_file_path):
            raise FileNotFoundError(f"Sentence transformer model file not found: {sentence_transformer_file_path}")

        # load the sentence transformer model
        model: SentenceTransformer = sentence_transformer_load_object(sentence_transformer_file_path)
        embeddings = model.encode(non_legacy_crm_df['log_message'].values)
        # train the model
        X = embeddings
        y = non_legacy_crm_df['target_label']
        # train the model
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self.model_trainer_config.model_trainer_test_train_split,
                                                            random_state=42)
        reg = LogisticRegression(max_iter=1000)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification report for the non-legacy crm data: {report}")
        # save the model
        save_object(self.model_trainer_config.model_trainer_model_file_path, reg)
        logger.info(f"Model saved successfully at: {self.model_trainer_config.model_trainer_model_file_path}")
        return self.model_trainer_config.model_trainer_model_file_path

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        tag: str = f"{self.class_name}::initiate_model_trainer"
        try:
            logger.info(f"{tag}::Initiating model training")
            # Load data
            if not os.path.exists(self.data_transformation_artifact.transformed_data_file_path):
                raise FileNotFoundError(f"{tag}::File not found: {self.data_transformation_artifact.transformed_data_file_path}")
            train_df = pd.read_csv(self.data_transformation_artifact.transformed_data_file_path)
            logger.info(f"{tag}::Data loaded successfully")

            # load the non-classified data
            if not os.path.exists(self.data_transformation_artifact.regex_none_classified_data_file_path):
                raise FileNotFoundError(f"{tag}::File not found: {self.data_transformation_artifact.regex_none_classified_data_file_path}")

            non_classified_df = pd.read_csv(self.data_transformation_artifact.regex_none_classified_data_file_path)
            # We are using the BERT model to classify the non-classified data, i.e., the data that is not classified by the regex model.

            """
            The non-classified data is the data that is not classified by the regex model.
            According to the data analysis, in the non-classified data the data belonging to the 'LegacyCRM' then there will be very few messages
            This comes from the domain knowledge that the 'LegacyCRM' is a very old system and the messages from this system are very few.
            The 'LegacyCRM' has only two categories 'Workflow Error' and 'Deprecation Warning'.
            Since there are so few messages we will use LLM for classification for the 'LegacyCRM' messages.
            For the other non-classified data we will be using the BERT model for classification.
            """
            # print the unique values in the target column of the non-classified data
            # print(non_classified_df['target_label'].value_counts()[non_classified_df['target_label'].value_counts() <= 5].index.tolist())
            # perform BERT classification on the non-classified data without the source 'LegacyCRM'
            # create a new dataframe with the non-classified data without the source 'LegacyCRM'
            non_legacy_crm_df = non_classified_df[non_classified_df['source'] != 'LegacyCRM']
            # check if the non-legacy crm dataframe is empty
            if non_legacy_crm_df.empty:
                logger.warning(f"{tag}::No data to classify for non-legacy crm")
            else:
                # perform BERT classification on the non-legacy crm data
                self.perform_bret_classification(non_legacy_crm_df)
            return ModelTrainerArtifact(self.model_trainer_config.model_trainer_model_file_path)
        except Exception as e:
            logger.error(f"{tag}::Error in loading data: {str(e)}")
            raise CustomException(e, sys)