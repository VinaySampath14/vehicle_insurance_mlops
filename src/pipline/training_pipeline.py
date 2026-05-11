import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging


class TrainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion stage.")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion stage complete. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation stage.")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation stage complete. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
        try:
            logging.info("========== Training Pipeline started ==========")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            # Next stages (transformation, trainer, …) will be added here
            logging.info("========== Training Pipeline completed ==========")
        except Exception as e:
            raise MyException(e, sys)
