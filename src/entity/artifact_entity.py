from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """Output of the DataIngestion component."""
    trained_file_path: str   # path to train.csv saved locally
    test_file_path: str      # path to test.csv saved locally


@dataclass
class DataValidationArtifact:
    """Output of the DataValidation component."""
    validation_status: bool          # True if data passed all schema checks
    valid_train_file_path: str       # path to validated train file
    valid_test_file_path: str        # path to validated test file
    invalid_train_file_path: str     # path to rejected train file (if any rows failed)
    invalid_test_file_path: str      # path to rejected test file (if any rows failed)
    drift_report_file_path: str      # path to the YAML drift report


@dataclass
class DataTransformationArtifact:
    """Output of the DataTransformation component."""
    transformed_object_file_path: str   # fitted preprocessor (.pkl) — needed at inference time
    transformed_train_file_path: str    # transformed train data as numpy array (.npy)
    transformed_test_file_path: str     # transformed test data as numpy array (.npy)


@dataclass
class ClassificationMetricArtifact:
    """Holds model performance metrics — reused by both trainer and evaluator."""
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    """Output of the ModelTrainer component."""
    trained_model_file_path: str                    # path to saved model.pkl
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    """
    Output of the ModelEvaluation component.

    is_model_accepted: False means the new model didn't beat production by the threshold,
                       so the pipeline stops here and doesn't push.
    improved_accuracy: how much better (or worse) the new model is vs production.
    """
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str         # path to whichever model won (new or prod)
    trained_model_path: str      # path to the newly trained model
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    """Output of the ModelPusher component."""
    bucket_name: str      # S3 bucket where model was pushed
    s3_model_path: str    # full S3 key/path of the pushed model
