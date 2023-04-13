import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler 
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

@dataclass
class DataTransform:
    data_transform_config = DataTransformConfig()

    def build_pipeline(self):
        try:
            logging.info('Build Pipeline Initiated')

            # Numerical and Categorical Columns

            cats_cols = [
                'cut', 
                'color', 
                'clarity'
                ]
            nums_cols = [
                'carat', 
                'depth', 
                'table', 
                'x', 
                'y', 
                'z'
                ]
            # Categories inside categorical columns

            cut_categories = [
                'Fair', 
                'Good', 
                'Very Good', 
                'Premium', 
                'Ideal'
                ]
            clarity_categories = [
                'I1', 
                'SI2', 
                'SI1', 
                'VS2', 
                'VS1', 
                'VVS2', 
                'VVS1', 
                'IF'
                ]
            color_categories = [
                'D', 
                'E', 
                'F', 
                'G', 
                'H', 
                'I', 
                'J'
                ]
            
            # Numerical Pipeline

            nums_pipe = Pipeline(
                steps = (
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaling', StandardScaler())
                )
            )

            logging.info('Numerical Pipeline Created')
            # Categorical Pipeline

            cats_pipe = Pipeline(
                steps = (
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories = [cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())
                )
            )

            logging.info('Categorical Pipeline Created')

            preprocessor = ColumnTransformer(
                transformers = [
                ('Num', nums_pipe, nums_cols),
                ('Cat', cats_pipe, cats_cols)
                ]
            )
            logging.info('Build Pipeline Successful')

            return preprocessor

        except Exception as e:
            logging.error('Build Pipeline Failed')
            raise CustomException(e, sys)
    

    def initiate_data_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            

            logging.info('Train Test Data Loaded Succesful')
            logging.info(f'Train DataFrame: \n{train_df.head(3).to_string()}')
            logging.info(f'Test DataFrame: \n{test_df.head(3).to_string()}')
            
            target_col_name = 'price'
            drop_columns = [target_col_name, 'id']

            features_train_data = train_df.drop(columns = drop_columns, axis = 1)
            target_train_data = train_df[target_col_name]

            features_test_data = test_df.drop(columns = drop_columns, axis = 1)
            target_test_data = test_df[target_col_name]


            preprocessor = self.build_pipeline()
            logging.info('Pipeline Loaded Successful')

            preprocessor.fit(features_train_data)

            transform_feature_train_data = preprocessor.transform(features_train_data)
            transform_feature_test_data = preprocessor.transform(features_test_data)

            logging.info('Data Transformation Successful')

            train_data = np.c_[transform_feature_train_data, np.array(target_train_data)]
            test_data = np.c_[transform_feature_test_data, np.array(target_test_data)]

            save_object(
                file_path = self.data_transform_config.preprocessor_obj_path,
                obj = preprocessor
            )

            return (
                train_data,
                test_data,
                self.data_transform_config.preprocessor_obj_path
            )

        except Exception as e:
            logging.error('Data Transform Failed')
            raise CustomException(e, sys)
        
        