#  NAMES
pipeline_name = "log_classifier"
artifact_dir: str = "artifacts"
# FEATURE CONSTANTS
x_feature_names = ['log_message']
y_target_feature = 'target_label'
# GENERAL CONSTANTS
data_file_folder_name="data"
data_file_name="synthetic_logs.csv"
# DATA INGESTION CONSTANTS
data_ingestion_dir_name: str = "data_ingestion"
data_ingestion_feature_store_dir_name: str = "feature_store"
data_ingestion_ingested_data_dir_name: str = "ingested"