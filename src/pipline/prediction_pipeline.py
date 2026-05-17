import sys
import pandas as pd
from src.entity.s3_estimator import S3Estimator
from src.constants import MODEL_BUCKET_NAME, MODEL_PUSHER_S3_KEY
from src.exception import MyException
from src.logger import logging


class PredictionPipeline:
    """Loads the production model from S3 and runs inference on new data."""

    def __init__(self):
        self.bucket_name = MODEL_BUCKET_NAME
        self.model_path = MODEL_PUSHER_S3_KEY

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("Loading production model from S3 for prediction.")
            estimator = S3Estimator(bucket_name=self.bucket_name, model_path=self.model_path)

            if not estimator.is_model_present():
                raise Exception(
                    f"No model found at s3://{self.bucket_name}/{self.model_path}. "
                    "Run the training pipeline first."
                )

            model = estimator.load_model()
            predictions = model.predict(dataframe)
            logging.info(f"Predictions generated for {len(dataframe)} records.")
            return predictions

        except Exception as e:
            raise MyException(e, sys)


class VehicleData:
    """Holds one row of user input and converts it to a DataFrame for prediction."""

    def __init__(
        self,
        gender: str,
        age: int,
        driving_license: int,
        region_code: float,
        previously_insured: int,
        vehicle_age: str,
        vehicle_damage: str,
        annual_premium: float,
        policy_sales_channel: float,
        vintage: int,
    ):
        self.gender = gender
        self.age = age
        self.driving_license = driving_license
        self.region_code = region_code
        self.previously_insured = previously_insured
        self.vehicle_age = vehicle_age
        self.vehicle_damage = vehicle_damage
        self.annual_premium = annual_premium
        self.policy_sales_channel = policy_sales_channel
        self.vintage = vintage

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            return pd.DataFrame([{
                "Gender": self.gender,
                "Age": self.age,
                "Driving_License": self.driving_license,
                "Region_Code": self.region_code,
                "Previously_Insured": self.previously_insured,
                "Vehicle_Age": self.vehicle_age,
                "Vehicle_Damage": self.vehicle_damage,
                "Annual_Premium": self.annual_premium,
                "Policy_Sales_Channel": self.policy_sales_channel,
                "Vintage": self.vintage,
            }])
        except Exception as e:
            raise MyException(e, sys)
