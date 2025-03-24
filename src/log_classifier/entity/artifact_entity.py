from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    # these are the inputs to the data ingestion pipeline
    train_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str