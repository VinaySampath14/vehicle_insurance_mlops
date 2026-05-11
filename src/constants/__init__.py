import os

# MongoDB
MONGODB_URL_KEY = "MONGODB_URL"
DATABASE_NAME   = "Proj1"
COLLECTION_NAME = "Proj1-Data"

# Pipeline
PIPELINE_NAME = "vehicle_insurance"
ARTIFACT_DIR  = "artifact"

# Data Ingestion
DATA_INGESTION_DIR                      = "data_ingestion"
DATA_INGESTION_FEATURE_STORE            = "feature_store"
DATA_INGESTION_INGESTED_DIR             = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
TRAIN_FILE_NAME                         = "train.csv"
TEST_FILE_NAME                          = "test.csv"
FILE_NAME                               = "data.csv"

# Data Validation
DATA_VALIDATION_DIR                     = "data_validation"
DATA_VALIDATION_VALID_DIR               = "validated"
DATA_VALIDATION_INVALID_DIR             = "invalid"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME  = "report.yaml"

# Data Transformation
DATA_TRANSFORMATION_DIR                     = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR    = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR  = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME              = "preprocessing.pkl"

# Model Trainer
MODEL_TRAINER_DIR                       = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR         = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME        = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float     = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH    = os.path.join("config", "model.yaml")

# Model Evaluation
MODEL_EVALUATION_DIR                            = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME                    = "report.yaml"
MODEL_BUCKET_NAME                               = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY                             = "model-registry"

# Model Pusher
MODEL_PUSHER_DIR             = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = os.path.join("saved_models")

# AWS
AWS_ACCESS_KEY_ID_ENV_KEY     = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME                   = "us-east-1"
