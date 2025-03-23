import os
import sys
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

    def initiate_data_ingestion(self, source_file_path: str = None) -> DataIngestionArtifact:
        tag: str = f"{self.class_name}::data_ingestion::"
        if source_file_path is None:
            source_file_path = self.config.get_data_file_path()
        else:
            # if the source file path is provided, update the config
            self.config.set_data_file_path(source_file_path)
            logger.info(f"{tag}::Data file path updated to {source_file_path}")

        if not os.path.exists(source_file_path):
            message:str = f"File {source_file_path} does not exist"
            logger.error(message)
            raise CustomException(message, sys)

        # write ingestion logic as needed
        return DataIngestionArtifact(data_file_path=source_file_path)

