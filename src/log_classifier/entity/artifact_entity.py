from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    # these are the inputs to the data ingestion pipeline
    train_file_path: str