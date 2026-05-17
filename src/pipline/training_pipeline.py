import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from src.exception import MyException
from src.logger import logging


class TrainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

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

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation stage.")
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation stage complete. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer stage.")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model trainer stage complete. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation stage.")
            model_evaluation = ModelEvaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_config=self.model_evaluation_config,
            )
            artifact = model_evaluation.initiate_model_evaluation()
            logging.info(f"Model evaluation stage complete. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            logging.info("Starting model pusher stage.")
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            artifact = model_pusher.initiate_model_pusher()
            logging.info(f"Model pusher stage complete. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
        try:
            logging.info("========== Training Pipeline started ==========")
            data_ingestion_artifact      = self.start_data_ingestion()
            data_validation_artifact     = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact       = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact    = self.start_model_evaluation(data_transformation_artifact, model_trainer_artifact)

            if model_evaluation_artifact.is_model_accepted:
                self.start_model_pusher(model_evaluation_artifact)
            else:
                logging.info("Model not accepted — skipping push to S3.")

            logging.info("========== Training Pipeline completed ==========")
        except Exception as e:
            raise MyException(e, sys)
