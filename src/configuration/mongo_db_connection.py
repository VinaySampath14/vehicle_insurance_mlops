import os
import pymongo
from src.constants import MONGODB_URL_KEY
from src.exception import MyException
from src.logger import logging
import sys

class MongoDBClient:
    client = None  # shared connection across all instances

    def __init__(self, database_name: str):
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.environ.get(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)

            self.client   = MongoDBClient.client
            self.database = self.client[database_name]
            logging.info(f"MongoDB connection established - database: {database_name}")

        except Exception as e:
            raise MyException(e, sys)
