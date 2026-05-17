import sys
import os
from sklearn.metrics import f1_score, precision_score, recall_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ClassificationMetricArtifact,
    DataTransformationArtifact,
)
from src.entity.s3_estimator import S3Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    write_yaml_file,
)


class ModelEvaluation:
    """
    Compares the newly trained model against the production model.
    Loads test data and generates metrics for both models.
    Decides whether the new model is good enough to push to production.
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(),
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def get_classification_metrics(y_true, y_pred) -> ClassificationMetricArtifact:
        """Calculate F1, Precision, Recall on predictions."""
        try:
            return ClassificationMetricArtifact(
                f1_score=f1_score(y_true, y_pred),
                precision_score=precision_score(y_true, y_pred),
                recall_score=recall_score(y_true, y_pred),
            )
        except Exception as e:
            raise MyException(e, sys)

    def get_best_model(self):
        """Load production model from S3. Returns None if no model exists yet (first run)."""
        try:
            logging.info("Attempting to load production model from S3...")
            s3_estimator = S3Estimator(
                bucket_name=self.model_evaluation_config.bucket_name,
                model_path=self.model_evaluation_config.s3_key_name,
            )
            if s3_estimator.is_model_present():
                model = s3_estimator.load_model()
                logging.info("Production model loaded from S3.")
                return model
            else:
                logging.info("No production model found on S3 — this is the first model.")
                return None
        except Exception as e:
            logging.warning(f"Could not load production model: {str(e)}")
            return None

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(">>>>>> Model Evaluation started <<<<<<")

            # Load test data (prepared by DataTransformation)
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info(f"Test data loaded: X shape {X_test.shape}, y shape {y_test.shape}")

            # Metrics for the newly trained model
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            trained_model_metrics = self.model_trainer_artifact.test_metric_artifact
            logging.info(
                f"Trained model test metrics - F1: {trained_model_metrics.f1_score:.4f} | "
                f"Precision: {trained_model_metrics.precision_score:.4f} | "
                f"Recall: {trained_model_metrics.recall_score:.4f}"
            )

            # Try to load production model
            production_model = self.get_best_model()

            if production_model is None:
                # First model — accept it by default
                logging.info("First model — automatically accepted for production")
                is_model_accepted = True
                improved_accuracy = 0.0
                best_model_path = self.model_trainer_artifact.trained_model_file_path
                best_model_metrics = trained_model_metrics
            else:
                # Compare: new model vs production model
                production_model_metrics = self.get_classification_metrics(
                    y_test, production_model.predict(X_test)
                )
                logging.info(
                    f"Production model test metrics - F1: {production_model_metrics.f1_score:.4f} | "
                    f"Precision: {production_model_metrics.precision_score:.4f} | "
                    f"Recall: {production_model_metrics.recall_score:.4f}"
                )

                # Calculate improvement
                improved_accuracy = (
                    trained_model_metrics.f1_score - production_model_metrics.f1_score
                )
                logging.info(
                    f"F1 score improvement: {improved_accuracy:.4f} "
                    f"(threshold: {self.model_evaluation_config.changed_threshold_score})"
                )

                # Decision: accept new model if improvement exceeds threshold
                if improved_accuracy >= self.model_evaluation_config.changed_threshold_score:
                    is_model_accepted = True
                    best_model_path = self.model_trainer_artifact.trained_model_file_path
                    best_model_metrics = trained_model_metrics
                    logging.info("✓ New model ACCEPTED — improvement exceeds threshold")
                else:
                    is_model_accepted = False
                    best_model_path = None  # Keep production model (not included in artifact)
                    best_model_metrics = production_model_metrics
                    logging.info("✗ New model REJECTED — insufficient improvement, keeping production model")

            # Write evaluation report
            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
            report = {
                "is_model_accepted": is_model_accepted,
                "improved_accuracy": float(improved_accuracy),
                "best_model_path": best_model_path,
                "trained_model_path": self.model_trainer_artifact.trained_model_file_path,
                "train_model_f1": float(self.model_trainer_artifact.train_metric_artifact.f1_score),
                "train_model_precision": float(self.model_trainer_artifact.train_metric_artifact.precision_score),
                "train_model_recall": float(self.model_trainer_artifact.train_metric_artifact.recall_score),
                "test_model_f1": float(trained_model_metrics.f1_score),
                "test_model_precision": float(trained_model_metrics.precision_score),
                "test_model_recall": float(trained_model_metrics.recall_score),
            }
            write_yaml_file(self.model_evaluation_config.report_name, report)
            logging.info(f"Evaluation report written to: {self.model_evaluation_config.report_name}")

            # Create artifact
            artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=best_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact,
                best_model_metric_artifact=best_model_metrics,
            )
            logging.info(f"Model Evaluation artifact: {artifact}")
            logging.info(">>>>>> Model Evaluation completed <<<<<<\n")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
