import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_object(file_path, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f'Success Object Saved at {file_path}')

    except Exception as e:
        logging.error(f'FAILED to Save Object at {file_path}')
        raise CustomException(e, sys)
    
def evalute_model(X_train, X_test, y_train, y_test, models) -> pd.DataFrame:
    try:
        report = []

        for model in models:
            MODEL = models[model]

            # Model Training
            MODEL.fit(X_train, y_train)

            # Predict Test Data
            y_pred = MODEL.predict(X_test)

            # Evaluation

            report.append([
                model,
                mean_absolute_error(y_test, y_pred),
                r2_score(y_test, y_pred),
                np.sqrt(mean_squared_error(y_test, y_pred))
            ])

        return pd.DataFrame(report, columns = ['ModelName', 'MAE', 'R2Score', 'RMSE'])

    except Exception as e:
        logging.error('FAILED to Train Model')
        raise CustomException(e, sys)
