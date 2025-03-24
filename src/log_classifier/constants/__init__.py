import os
#  NAMES
pipeline_name = "log_classifier"
artifact_dir: str = "artifacts"

# FEATURE CONSTANTS
x_feature_names = 'log_message'
y_target_feature = 'target_label'
cluster_label = 'cluster_label'
regex_label = 'regex_label'

# GENERAL CONSTANTS
data_file_folder_name="data"
data_file_name="synthetic_logs.csv"
train_file_name: str = "train_data.csv"
schema_file_path: str = os.path.join("data_schema", "schema.yaml")

# DATA INGESTION CONSTANTS
data_ingestion_dir_name: str = "data_ingestion"
data_ingestion_feature_store_dir_name: str = "feature_store"
data_ingestion_ingested_data_dir_name: str = "ingested"

# DATA VALIDATION CONSTANTS
data_validation_dir_name: str = "data_validation"
data_validation_valid_dir: str = "validated"

# DATA TRANSFORMATION CONSTANTS
data_transformation_dir_name: str = "data_transformation"
data_transformation_embeddings_dir: str = "embeddings"
data_transformation_embeddings_file_name: str = "embeddings.npy"
data_transformation_data_dir: str = "transformed"
data_transformation_regex_none_classified: str = "classified_none_" + train_file_name
data_transformation_regex_classified: str = "classified_" + train_file_name
sentence_transformer_model_name: str = "all-mpnet-base-v2"
# these dbscan values need to be tuned by trail and error
dbscan_eps=0.2
dbscan_min_samples=1
dbscan_metric='cosine'

# MODEL TRAINING CONSTANTS
model_trainer_dir_name: str = "model_training"