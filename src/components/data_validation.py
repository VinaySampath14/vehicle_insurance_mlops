import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constants import SCHEMA_FILE_PATH
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks that the dataframe has the same number of columns as defined in schema.yaml.
        A mismatch means a column was added or dropped upstream — that breaks all downstream code.
        """
        try:
            expected = len(self._schema_config["columns"])
            actual = len(dataframe.columns)
            if actual != expected:
                logging.warning(f"Column count mismatch: expected {expected}, got {actual}")
                return False
            return True
        except Exception as e:
            raise MyException(e, sys)

    def is_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks that every column defined in schema.yaml is present.
        A missing column would silently produce NaN-filled features if not caught here.
        """
        try:
            missing = [
                col for col in self._schema_config["columns"]
                if col not in dataframe.columns
            ]
            if missing:
                logging.warning(f"Missing columns: {missing}")
                return False
            return True
        except Exception as e:
            raise MyException(e, sys)

    def detect_dataset_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """
        Runs the KS test (Kolmogorov-Smirnov) on every numerical column.

        MLOps reasoning from notebook:
          In the notebook we saw Age, Annual_Premium, Vintage are the key numerical features.
          If their distributions shift between train and test (or between runs), the model's
          learned decision boundaries won't generalise.

        KS test: p-value < threshold means the two samples likely come from different
        distributions → drift detected. We log it but treat it as a warning, not a blocker.
        The full report is written to report.yaml for the data team to inspect.
        """
        try:
            status = True
            report = {}

            for column in base_df.columns:
                if base_df[column].dtype in ["float64", "int64"]:
                    d_train = base_df[column]
                    d_test = current_df[column]
                    ks_result = ks_2samp(d_train, d_test)
                    drift_found = ks_result.pvalue < threshold
                    if drift_found:
                        status = False
                    report[column] = {
                        "p_value": float(ks_result.pvalue),
                        "drift_status": bool(drift_found),
                    }

            drift_report_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=drift_report_path, content=report, replace=True)
            return status

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info(">>>>>> Data Validation started <<<<<<")
            validation_error_msg = ""

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Column count check
            for name, df in [("train", train_df), ("test", test_df)]:
                if not self.validate_number_of_columns(df):
                    validation_error_msg += f"{name} dataset: column count mismatch. "

            # Column existence check
            for name, df in [("train", train_df), ("test", test_df)]:
                if not self.is_column_exist(df):
                    validation_error_msg += f"{name} dataset: required columns missing. "

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                logging.info(f"Drift detection complete. No drift: {drift_status}")
            else:
                logging.warning(f"Validation failed: {validation_error_msg}")

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)

            if validation_status:
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)
            else:
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data Validation artifact: {artifact}")
            logging.info(">>>>>> Data Validation completed <<<<<<\n")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
