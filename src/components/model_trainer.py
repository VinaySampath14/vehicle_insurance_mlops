import sys
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.entity.estimator import VehicleInsuranceModel
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object, read_yaml_file


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise MyException(e, sys)

    def get_model_from_config(self, scale_pos_weight: float) -> XGBClassifier:
        """
        Reads model.yaml and builds XGBClassifier.
        scale_pos_weight is passed in (not in YAML) because it's calculated
        from the actual class distribution in training data, not a fixed value.
        """
        try:
            config = read_yaml_file(self.model_trainer_config.model_config_file_path)
            params = config["model"]["params"]
            params["scale_pos_weight"] = scale_pos_weight
            model = XGBClassifier(**params)
            logging.info(f"Model loaded from config: {model}")
            return model
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def get_classification_metrics(y_true, y_pred) -> ClassificationMetricArtifact:
        try:
            return ClassificationMetricArtifact(
                f1_score=f1_score(y_true, y_pred),
                precision_score=precision_score(y_true, y_pred),
                recall_score=recall_score(y_true, y_pred),
            )
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(">>>>>> Model Trainer started <<<<<<")

            # Load transformed numpy arrays — last column is the target
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr  = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            # Calculate class imbalance ratio for XGBoost
            # scale_pos_weight = negatives / positives — tells XGBoost how much
            # more to penalise errors on the minority class
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            scale_pos_weight = neg / pos
            logging.info(f"Class ratio - Negative: {int(neg)} | Positive: {int(pos)} | scale_pos_weight: {scale_pos_weight:.2f}")

            # Train
            model = self.get_model_from_config(scale_pos_weight=scale_pos_weight)
            logging.info("Training model...")
            model.fit(X_train, y_train)

            # Evaluate on both splits
            # Why check train metrics too? To detect overfitting.
            # A huge gap between train F1 and test F1 means the model memorised training data.
            train_metrics = self.get_classification_metrics(y_train, model.predict(X_train))
            test_metrics  = self.get_classification_metrics(y_test,  model.predict(X_test))

            logging.info(f"Train metrics - F1: {train_metrics.f1_score:.4f} | Precision: {train_metrics.precision_score:.4f} | Recall: {train_metrics.recall_score:.4f}")
            logging.info(f"Test  metrics - F1: {test_metrics.f1_score:.4f}  | Precision: {test_metrics.precision_score:.4f}  | Recall: {test_metrics.recall_score:.4f}")

            # Threshold check — if test F1 is below minimum, reject this model
            if test_metrics.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model F1 score {test_metrics.f1_score:.4f} is below the expected threshold "
                    f"{self.model_trainer_config.expected_accuracy}. Model not saved."
                )

            # Load the preprocessor and bundle it with the model into one object
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            vehicle_model = VehicleInsuranceModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, vehicle_model)
            logging.info(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )
            logging.info(f"Model Trainer artifact: {artifact}")
            logging.info(">>>>>> Model Trainer completed <<<<<<\n")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
