from datetime import datetime
from src.log_classifier.constants import pipeline_name, data_file_folder_name, data_file_name


class PipelineConfig:
    def __init__(self, timestamp=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
        self.class_name = self.__class__.__name__
        self.timestamp = timestamp
        self.pipeline = pipeline_name
        self.data_file_folder_name = data_file_folder_name
        self.data_file_name = data_file_name