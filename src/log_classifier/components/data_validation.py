import os
import sys
import pandas as pd
from src.log_classifier.constants import schema_file_path
from src.log_classifier.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.log_classifier.entity.config_entity import DataValidationConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger
from src.log_classifier.utils.utils import read_yaml, save_dataframe


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.class_name = self.__class__.__name__
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml(schema_file_path)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, data: pd.DataFrame) -> bool:
        tag: str = f"{self.class_name}::validate_number_of_columns::"
        try:
            numbers_of_data_columns = len(data.columns)
            numbers_of_data_columns_in_schema = len(self._schema_config['columns'])
            if numbers_of_data_columns == numbers_of_data_columns_in_schema:
                logger.info(f"{tag}::Number of columns in data is same as schema: {numbers_of_data_columns}")
                return True
            else:
                logger.error(f"{tag}::Number of columns in data is different from schema")
                logger.error(
                    f"{tag}::Number of columns in data: {numbers_of_data_columns} and schema: {numbers_of_data_columns_in_schema} are not the same")
                return False
        except Exception as e:
            logger.error(f"{tag}::Error validating number of columns: {e}")
            raise CustomException(e, sys)

    def validate_column_names(self, data: pd.DataFrame) -> bool:
        tag: str = f"{self.class_name}::validate_column_names::"
        try:
            numerical_columns = self._schema_config['columns']
            data_columns_list = list(data.columns)
            missing_columns = [col for col in numerical_columns if col not in data_columns_list]
            if missing_columns:
                logger.error(f"{tag}::Missing numerical columns in data: {missing_columns}")
                return False
            logger.info(f"{tag}::All columns are validated")
            return True
        except Exception as e:
            logger.error(f"{tag}::Error validating numerical columns: {e}")
            raise CustomException(e, sys)

    def validate_data(self, train_data: pd.DataFrame) -> bool:
        tag: str = f"{self.class_name}::validate_data::"

        logger.info(f"{tag}::Validating number of columns in train data")
        is_columns_numbers_same_train = self.validate_number_of_columns(train_data)
        if not is_columns_numbers_same_train:
            logger.error(f"{tag}::Data validation failed. Train data has different number of columns")

        logger.info(f"{tag}::Validating column names in train data")
        is_numerical_columns_same_train = self.validate_column_names(train_data)
        if not is_numerical_columns_same_train:
            logger.error(f"{tag}::Data validation failed. Train data numerical columns are not same as schema")
        return is_columns_numbers_same_train and is_numerical_columns_same_train

    def initiate_data_validation(self) -> DataValidationArtifact:
        tag: str = f"{self.class_name}::initiate_data_validation::"
        try:
            logger.info(f"{tag}::Initiating data validation")
            # get train and test file path
            train_file_path = self.data_ingestion_artifact.train_file_path
            if not train_file_path:
                raise CustomException(f"Train file path {train_file_path} is empty", sys)

            if not os.path.exists(train_file_path):
                raise CustomException(f"Train file {train_file_path} does not exist", sys)

            # read data from train and test file
            train_data = DataValidation.read_data(train_file_path)

            # validate train and test data
            data_status = self.validate_data(train_data)
            if not data_status:
                logger.error(f"{tag}::Data validation failed.")
                logger.error(f"{tag}::Data columns are not same as schema")
                # raise CustomException("Data validation failed", sys)
            logger.info(f"{tag}::Data validation for columns completed successfully")

            # save test and train data
            valid_data_dir = self.data_validation_config.valid_data_dir
            # create directory if not exists
            os.makedirs(valid_data_dir, exist_ok=True)
            logger.info(f"{tag}::Folder created: {valid_data_dir}")
            save_dataframe(train_data, self.data_validation_config.valid_train_file_path, f"{tag}::validated training data export")
            logger.info(f"{tag}::Validated data saved to csv successfully")

            # create data validation artifact
            status = data_status
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path)
            return data_validation_artifact
        except Exception as e:
            logger.error(f"{tag}::Error running the data validation pipeline: {e}")
            raise CustomException(e, sys)