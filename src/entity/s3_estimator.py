import sys
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging


class S3Estimator:
    """
    Handles saving and loading the production model to/from S3.
    Used by ModelEvaluation (to load prod model) and ModelPusher (to save new model).
    """

    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()

    def is_model_present(self) -> bool:
        try:
            return self.s3.s3_key_path_available(self.bucket_name, self.model_path)
        except Exception as e:
            raise MyException(e, sys)

    def load_model(self):
        try:
            logging.info(f"Loading production model from s3://{self.bucket_name}/{self.model_path}")
            return self.s3.load_model(self.model_path, self.bucket_name)
        except Exception as e:
            raise MyException(e, sys)

    def save_model(self, from_file: str, to_file: str = None):
        try:
            destination = to_file if to_file else self.model_path
            logging.info(f"Saving model to s3://{self.bucket_name}/{destination}")
            self.s3.upload_file(
                from_filename=from_file,
                to_filename=destination,
                bucket_name=self.bucket_name,
                remove=False,
            )
        except Exception as e:
            raise MyException(e, sys)
