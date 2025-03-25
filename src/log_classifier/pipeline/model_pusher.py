import sys

from src.log_classifier.components.model_pusher import ModelPusher
from src.log_classifier.entity.artifact_entity import ModelTrainerArtifact
from src.log_classifier.entity.config_entity import ModelPusherConfig
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger

STAGE_NAME: str = "Model Pusher Pipeline"
class ModelPusherTrainingPipeline:
    def __init__(self, model_trainer_artifact:  ModelTrainerArtifact):
        self.class_name = self.__class__.__name__
        self.stage_name = STAGE_NAME
        self.model_trainer_artifact = model_trainer_artifact

    def model_pusher(self) -> None:
        tag: str = f"[{self.class_name}][{self.model_pusher.__name__}]::"
        try:
            logger.info(f"{tag}::Model pusher pipeline started")

            model_pusher_config: ModelPusherConfig = ModelPusherConfig()
            logger.info(f"{tag}::Model pusher configuration obtained")

            model_pusher = ModelPusher(config=model_pusher_config, model_trainer_artifact=self.model_trainer_artifact)
            logger.info(f"{tag}::Model pusher object created")

            logger.info(f"{tag}::Running the model pusher pipeline")
            model_pusher.push()
            logger.info(f"{tag}::Model pusher pipeline completed")
        except Exception as e:
            logger.error(f"{tag}::Error running the model training pipeline: {e}")
            raise CustomException(e, sys)