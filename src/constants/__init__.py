import os

# MongoDB
MONGODB_URL_KEY = "MONGODB_URL"
DATABASE_NAME   = "Proj1"
COLLECTION_NAME = "Proj1-Data"

# Pipeline artifacts root
ARTIFACT_DIR = "artifact"

# Data Ingestion
DATA_INGESTION_DIR             = "data_ingestion"
DATA_INGESTION_FEATURE_STORE   = "feature_store"
DATA_INGESTION_INGESTED_DIR    = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
TRAIN_FILE_NAME                = "train.csv"
TEST_FILE_NAME                 = "test.csv"
FILE_NAME                      = "data.csv"
