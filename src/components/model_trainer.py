import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os
import sys
from src.utils import save_object, evalute_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

@dataclass
class ModelTrainer:
    model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_data, test_data):
        try:
            
            logging.info('Splitting Dependent and Independent Variable Train-Test Data')

            X_train, X_test, y_train, y_test = (
                train_data[:, :-1],
                test_data[:, :-1],
                train_data[:, -1],
                test_data[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            model_report = evalute_model(
                X_train,
                X_test,
                y_train,
                y_test,
                models
            )

            logging.info(f'Model Report: \n{model_report}')
            print('Model Reports:\n', model_report, '\n\n')


            best_model_name = model_report.sort_values(by = 'R2Score', ascending = False).iloc[0,0]
            print('Best Model:\n', model_report[model_report.ModelName == best_model_name], '\n\n')
            best_model = models[best_model_name]

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            logging.error('FAILED Model Training')
            raise CustomException(e, sys)