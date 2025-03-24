import sys

from src.log_classifier.components.data_ingestion import DataIngestion
from src.log_classifier.config.configuration import TrainingPipelineConfig
from src.log_classifier.entity.artifact_entity import DataIngestionArtifact
from src.log_classifier.entity.config_entity import DataIngestionConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger

STAGE_NAME: str = "Data Ingestion Pipeline"
class DataIngestionTrainingPipeline:
    def __init__(self):
        self.class_name = self.__class__.__name__
        self.stage_name = STAGE_NAME

    def data_ingestion(self) -> DataIngestionArtifact:
        tag: str = f"{self.class_name}::data_ingestion::"
        try:
            config: TrainingPipelineConfig = TrainingPipelineConfig()
            logger.info(f"{tag}::Configuration object created")

            data_ingestion_config: DataIngestionConfig = DataIngestionConfig(config)
            logger.info(f"{tag}::Data ingestion configuration obtained")

            data_ingestion: DataIngestion = DataIngestion(data_ingestion_config)
            logger.info(f"{tag}::Data ingestion object created")

            logger.info(f"{tag}::Running the data ingestion pipeline")

            data_ingestion_artifact: DataIngestionArtifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            logger.info(f"{tag}::Data ingestion completed successfully")
            return data_ingestion_artifact
        except Exception as e:
            logger.error(f"{tag}::Error running the data ingestion pipeline: {e}")
            raise CustomException(e, sys)