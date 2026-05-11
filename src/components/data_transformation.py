import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Builds the preprocessing pipeline.

        Notebook → production decisions:
        - StandardScaler on Age, Vintage: both are roughly normal distributions
        - MinMaxScaler on Annual_Premium: right-skewed with outliers up to 540k,
          MinMaxScaler handles this better than StandardScaler
        - OrdinalEncoder on Gender: Female=0, Male=1 — matches notebook mapping exactly
        - OneHotEncoder on Vehicle_Age, Vehicle_Damage: categorical with no natural order,
          drop='first' avoids the dummy variable trap (multicollinearity)
        - remainder='passthrough': Driving_License, Previously_Insured, Region_Code,
          Policy_Sales_Channel are already numeric — no transformation needed
        """
        try:
            standard_cols  = ["Age", "Vintage"]
            minmax_cols    = ["Annual_Premium"]
            ordinal_cols   = ["Gender"]
            ohe_cols       = ["Vehicle_Age", "Vehicle_Damage"]

            standard_pipeline = Pipeline([("scaler", StandardScaler())])
            minmax_pipeline   = Pipeline([("scaler", MinMaxScaler())])
            ordinal_pipeline  = Pipeline([
                ("encoder", OrdinalEncoder(categories=[["Female", "Male"]]))
            ])
            ohe_pipeline = Pipeline([
                ("encoder", OneHotEncoder(drop="first", sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("standard_scaler", standard_pipeline, standard_cols),
                    ("minmax_scaler",   minmax_pipeline,   minmax_cols),
                    ("ordinal_encoder", ordinal_pipeline,  ordinal_cols),
                    ("ohe_encoder",     ohe_pipeline,      ohe_cols),
                ],
                remainder="passthrough",
            )

            return Pipeline([("preprocessor", preprocessor)])

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(">>>>>> Data Transformation started <<<<<<")

            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df  = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Drop id — unique identifier, zero predictive value
            train_df = train_df.drop(columns=["id"])
            test_df  = test_df.drop(columns=["id"])

            # Separate features and target
            target = "Response"
            X_train, y_train = train_df.drop(columns=[target]), train_df[target]
            X_test,  y_test  = test_df.drop(columns=[target]),  test_df[target]

            # Fit preprocessor on train only, transform both
            # Key rule: never fit on test data — that would leak test distribution into the model
            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(X_train)

            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed  = preprocessor.transform(X_test)

            # Combine features + target back into one array for easy saving
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr  = np.c_[X_test_transformed,  np.array(y_test)]

            logging.info(f"Transformed train shape: {train_arr.shape} | test shape: {test_arr.shape}")

            # Save transformed arrays and the fitted preprocessor object
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,  test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data Transformation artifact: {artifact}")
            logging.info(">>>>>> Data Transformation completed <<<<<<\n")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
