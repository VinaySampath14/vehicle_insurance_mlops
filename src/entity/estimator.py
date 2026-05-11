import sys
from sklearn.pipeline import Pipeline
from src.exception import MyException
from src.logger import logging


class VehicleInsuranceModel:
    """
    Wraps the trained sklearn model.
    Saving this object means one file contains everything needed to make predictions.
    """

    def __init__(self, preprocessor: Pipeline, model: object):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, x):
        try:
            x_transformed = self.preprocessor.transform(x)
            return self.model.predict(x_transformed)
        except Exception as e:
            raise MyException(e, sys)
