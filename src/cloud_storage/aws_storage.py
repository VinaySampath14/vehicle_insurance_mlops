import os
import sys
import pickle
from io import StringIO

from src.configuration.aws_connection import S3Client
from src.exception import MyException
from src.logger import logging


class SimpleStorageService:

    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            return any(True for _ in bucket.objects.filter(Prefix=s3_key))
        except Exception as e:
            raise MyException(e, sys)

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        try:
            logging.info(f"Uploading {from_filename} → s3://{bucket_name}/{to_filename}")
            self.s3_client.upload_file(from_filename, bucket_name, to_filename)
            if remove:
                os.remove(from_filename)
            logging.info("Upload complete.")
        except Exception as e:
            raise MyException(e, sys)

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None):
        try:
            local_path = os.path.join(model_dir, model_name) if model_dir else model_name
            os.makedirs(os.path.dirname(local_path), exist_ok=True) if os.path.dirname(local_path) else None
            logging.info(f"Downloading s3://{bucket_name}/{model_name} → {local_path}")
            self.s3_resource.Bucket(bucket_name).download_file(model_name, local_path)
            with open(local_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise MyException(e, sys)

    def upload_df_as_csv(self, df, bucket_filename: str, bucket_name: str):
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            self.s3_resource.Object(bucket_name, bucket_filename).put(Body=csv_buffer.getvalue())
            logging.info(f"DataFrame uploaded to s3://{bucket_name}/{bucket_filename}")
        except Exception as e:
            raise MyException(e, sys)
