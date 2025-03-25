import os
import sys

from src.log_classifier.entity.artifact_entity import ModelTrainerArtifact
from src.log_classifier.entity.config_entity import ModelPusherConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger
from src.log_classifier.utils.utils import copy_file_with_validation


class ModelPusher:
    def __init__(self, config: ModelPusherConfig,
                 model_trainer_artifact:  ModelTrainerArtifact):
        """Initialize the ModelPusher with AWS clients."""
        self.class_name = self.__class__.__name__
        self.config = config
        self.model_trainer_artifact = model_trainer_artifact

    def push(self):
        """Push the model to the model final folder."""
        tag:str = f"{self.class_name}::push"
        try:
            source = self.model_trainer_artifact.logistic_regression_model_file_path
            destination = self.config.model_pusher_dir_path
            # check if the source file exists
            if not os.path.exists(source):
                raise FileNotFoundError(f"Model file not found at {source}")

            # check if the destination folder exists
            if not os.path.exists(destination):
                os.makedirs(destination, exist_ok=True)

            # copy the model file to the destination folder
            logger.info(f"{tag}::Copying the model file from {source} to the destination folder {destination}")
            copy_file_with_validation(source, destination, [".pkl"])
            return destination
        except Exception as e:
            raise CustomException(f"{tag}::Error in pushing the model: {str(e)}", sys)