import sys
from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import S3Estimator
from src.exception import MyException
from src.logger import logging


class ModelPusher:

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig = ModelPusherConfig(),
    ):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.s3 = SimpleStorageService()

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info(">>>>>> Model Pusher started <<<<<<")

            best_model_path = self.model_evaluation_artifact.best_model_path
            bucket_name = self.model_pusher_config.bucket_name
            s3_key = self.model_pusher_config.s3_key_name

            logging.info(f"Pushing model: {best_model_path} → s3://{bucket_name}/{s3_key}")

            s3_estimator = S3Estimator(bucket_name=bucket_name, model_path=s3_key)
            s3_estimator.save_model(from_file=best_model_path)

            artifact = ModelPusherArtifact(bucket_name=bucket_name, s3_model_path=s3_key)
            logging.info(f"Model pushed successfully. Artifact: {artifact}")
            logging.info(">>>>>> Model Pusher completed <<<<<<\n")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
