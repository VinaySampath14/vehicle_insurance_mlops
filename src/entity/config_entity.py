import os
from dataclasses import dataclass
from datetime import datetime

from src.constants import *

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    """Root config — every other config derives its artifact_dir from this."""
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, PIPELINE_NAME, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    """
    Paths for raw data pulled from MongoDB and the resulting train/test split files.

    feature_store  → raw dump from MongoDB (full dataset, never split)
    ingested/      → train.csv and test.csv after the split
    """
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_INGESTION_DIR
    )
    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE, FILE_NAME
    )
    training_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME
    )
    testing_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
    )
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = COLLECTION_NAME


@dataclass
class DataValidationConfig:
    """
    Paths for the validation step.

    valid/invalid dirs hold copies of data that passed or failed schema checks.
    drift_report_file_path is a YAML report written after statistical drift detection.
    """
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR
    )
    valid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_VALID_DIR)
    invalid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_INVALID_DIR)
    valid_train_file_path: str = os.path.join(valid_data_dir, TRAIN_FILE_NAME)
    valid_test_file_path: str = os.path.join(valid_data_dir, TEST_FILE_NAME)
    invalid_train_file_path: str = os.path.join(invalid_data_dir, TRAIN_FILE_NAME)
    invalid_test_file_path: str = os.path.join(invalid_data_dir, TEST_FILE_NAME)
    drift_report_file_path: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
    )


@dataclass
class DataTransformationConfig:
    """
    Paths for the transformation step.

    transformed/ holds numpy arrays (.npy) of the scaled/encoded data.
    transformed_object/ holds the fitted preprocessor (scaler + encoder) as a .pkl
    so we can apply the exact same transformations at prediction time.
    """
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR
    )
    transformed_train_file_path: str = os.path.join(
        data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAIN_FILE_NAME
    ).replace(".csv", ".npy")
    transformed_test_file_path: str = os.path.join(
        data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TEST_FILE_NAME
    ).replace(".csv", ".npy")
    transformed_object_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCESSING_OBJECT_FILE_NAME,
    )


@dataclass
class ModelTrainerConfig:
    """
    Paths and thresholds for model training.

    expected_accuracy is the minimum score a freshly trained model must beat.
    If it doesn't, we don't even bother comparing with the production model.
    model_config_file_path points to config/model.yaml where hyperparameters live.
    """
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR
    )
    trained_model_file_path: str = os.path.join(
        model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME
    )
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


@dataclass
class ModelEvaluationConfig:
    """
    Config for comparing the newly trained model against the current production model on S3.

    changed_threshold_score: the new model must be at least this much better to replace prod.
    bucket_name / s3_key_name: location of the production model in AWS S3.
    """
    model_evaluation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR
    )
    report_name: str = os.path.join(model_evaluation_dir, MODEL_EVALUATION_REPORT_NAME)
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_key_name: str = MODEL_PUSHER_S3_KEY


@dataclass
class ModelPusherConfig:
    """
    Config for pushing an accepted model to AWS S3.

    saved_models/ is a local registry of all pushed models (useful for rollback).
    S3 bucket and key define where the model lives in production.
    """
    model_pusher_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR
    )
    saved_model_path: str = MODEL_PUSHER_SAVED_MODEL_DIR
    bucket_name: str = MODEL_BUCKET_NAME
    s3_key_name: str = MODEL_PUSHER_S3_KEY
