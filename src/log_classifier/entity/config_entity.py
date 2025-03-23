import os
from src.log_classifier.config.configuration import PipelineConfig


class DataIngestionConfig:
    def __init__(self, pipeline_config: PipelineConfig):
        self.class_name = self.__class__.__name__
        self.data_file_path = os.path.join(pipeline_config.data_file_folder_name, pipeline_config.data_file_name)

    def __str__(self):
        return f"{self.class_name}({self.data_file_path})"

    def get_data_file_path(self):
        return self.data_file_path

    def set_data_file_path(self, data_file_path):
        self.data_file_path = data_file_path
        return self.data_file_path