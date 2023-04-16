import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.utils import load_object
import pandas as pd
 

@dataclass
class PredictPipeline:

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            scaled_data = preprocessor.transform(features)

            prediction = model.predict(scaled_data)
            return prediction
        
        except Exception as e:
            logging.error('FAILED Prediction Pipeline')
            raise CustomException(e, sys)
        
@dataclass
class CustomData:
    carat:float
    depth:float
    table:float
    x:float
    y:float
    z:float
    cut:str
    color:str
    clarity:str

    def get_data_as_dataframe(self):
        try:
            col_names = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
            custom_data_input = [[self.carat, self.depth, self.table, self.x, self.y, self.z, self.cut, self.color, self.clarity]]


            data = pd.DataFrame(custom_data_input, columns = col_names)
            logging.info('DataFrame Gathered')
            return data
        except Exception as e:
            logging.error('FAILED DataFrame Gathered')
            raise CustomException(e, sys)
    
