import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException


class InsuranceData:

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str) -> pd.DataFrame:
        try:
            collection = self.mongo_client.database[collection_name]
            df = pd.DataFrame(list(collection.find()))

            # MongoDB adds an '_id' column — drop it
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            # Replace 'na' strings with actual NaN
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise MyException(e, sys)
