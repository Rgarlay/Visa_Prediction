import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from ..logger import logging
import os
import sys
from dataclasses import dataclass
from ..exception import CustomException
from ..utils import save_object,remove_outliers


@dataclass
class DataTransform:
    preprocessor_obj_file_path = os.path.join('archieve', 'preprocessor.pkl')  

class DataTransformation:
    def __init__(self):
        self.transformation = DataTransform()

    def transformations(self):
        try:
            logging.info("Starting transformation pipeline creation...")

            num_cols = ['no_of_employees', 'yr_of_estab', 'prevailing_wage']
            obj_cols = ['continent', 'education_of_employee', 'region_of_employment', 'unit_of_wage']
            binary_var = ['has_job_experience', 'requires_job_training', 'full_time_position']

            logging.info("Column names classified into numerical, categorical, and binary columns.")

            # Mapping binary columns
            map_binary_objs = Pipeline(steps=[
                ('label_encoding', OrdinalEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Outlier removal for numerical columns
            numerical_remove = Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Processing multiclass categorical features
            multiclass_obj = Pipeline(steps=[
                ('impute', SimpleImputer(strategy="most_frequent")),
                ('onehot', OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Combine all transformers
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', numerical_remove, num_cols),
                ('binary_cols', map_binary_objs, binary_var),
                ('multiclass_obj', multiclass_obj, obj_cols)
            ])

            logging.info("All transformation pipelines combined into ColumnTransformer.")

            return preprocessor
        
        except Exception as e:
            logging.error(f"Error in creating transformation pipeline: {e}")
            raise CustomException(e, sys)
        
    def data_transform_initiate(self, train_path, test_path):
        try:
            logging.info(f"Reading training data from {train_path} and test data from {test_path}.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data files successfully read.")

            preprocessing_obj = self.transformations()

            target_column_name = 'case_status'
            num_cols = ['no_of_employees', 'yr_of_estab', 'prevailing_wage']

            train_df_without_outliers = remove_outliers(train_df)
            test_df_without_outliers = remove_outliers(test_df)

            logging.info("Outliers removed from both datasets.")

            # Split the train dataset into features and target variable
            logging.info("Splitting the training dataset into input features and target variable.")

            input_features_train_df = train_df_without_outliers.drop(columns=[target_column_name], axis=1)
            target_features_train_df = train_df_without_outliers[target_column_name]

            # Split the test dataset into features and target variable
            logging.info("Splitting the testing dataset into input features and target variable.")

            input_features_test_df = test_df_without_outliers.drop(columns=[target_column_name], axis=1)
            target_features_test_df = test_df_without_outliers[target_column_name]


            logging.info("Applying preprocessing transformation on training and testing features...")
            input_train_features_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_test_features_arr = preprocessing_obj.transform(input_features_test_df)
            logging.info("Transformation applied successfully.")

            train_arr = np.c_[input_train_features_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_test_features_arr, np.array(target_features_test_df)]
            logging.info("Transformed data combined with target variables.")

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.transformation.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info(f"Preprocessing object saved at {self.transformation.preprocessor_obj_file_path}")
            return (train_arr, test_arr, self.transformation.preprocessor_obj_file_path)

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e,sys)
