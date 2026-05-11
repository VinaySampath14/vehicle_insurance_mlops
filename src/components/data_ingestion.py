import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.proj1_data import InsuranceData
from src.exception import MyException
from src.logger import logging


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Pulls the full dataset from MongoDB and saves it as a CSV to the feature store.
        The feature store is the raw, untouched copy — nothing is transformed here.
        """
        try:
            logging.info("Exporting data from MongoDB to feature store.")
            insurance_data = InsuranceData()
            dataframe = insurance_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False, header=True)
            logging.info(f"Data saved to feature store: {feature_store_path}  shape={dataframe.shape}")

            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Splits the raw dataframe into train and test CSVs.
        We do the split here (not in transformation) so that validation can check
        both splits independently before any preprocessing is applied.
        """
        try:
            logging.info("Splitting data into train and test sets.")
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=dataframe["Response"],
            )

            for path, split in [
                (self.data_ingestion_config.training_file_path, train_set),
                (self.data_ingestion_config.testing_file_path, test_set),
            ]:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                split.to_csv(path, index=False, header=True)

            logging.info(
                f"Train shape: {train_set.shape}  |  Test shape: {test_set.shape}"
            )

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the full ingestion flow and returns an artifact with
        the file paths that the next component (DataValidation) will consume.
        """
        try:
            logging.info(">>>>>> Data Ingestion started <<<<<<")
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe)

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data Ingestion artifact: {artifact}")
            logging.info(">>>>>> Data Ingestion completed <<<<<<\n")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
