from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    # these are the inputs to the data ingestion pipeline
    train_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str

@dataclass
class DataTransformationArtifact:
    model_embeddings_file_path: str
    transformed_data_file_path: str
    regex_none_classified_data_file_path: str
    regex_classified_data_file_path: str
    sentence_transformer_file_path: str

@dataclass
class ModelTrainerArtifact:
    logistic_regression_model_file_path: str

@dataclass
class ModelPusherArtifact:
    model_file_path: str