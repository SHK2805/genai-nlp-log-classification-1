import os
from src.log_classifier.config.configuration import PipelineConfig


class DataIngestionConfig:
    def __init__(self, pipeline_config: PipelineConfig):
        self.class_name = self.__class__.__name__
        self.data_file_path = os.path.join(pipeline_config.data_file_folder_name, pipeline_config.data_file_name)