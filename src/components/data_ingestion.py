import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransform
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the data ingestion component

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


@dataclass
class DataIngestion:
    ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method Start')
        try:
            data = pd.read_csv(os.path.join('notebooks', 'data', 'gemstone.csv'))
            logging.info('Dataset Loading Successful')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            data.to_csv(self.ingestion_config.raw_data_path, index = False)

            # Splitting Data in training and testing
            logging.info('Train Test Split')
            train_data, test_data = train_test_split(data, test_size = 0.3, random_state = 1)

            # Saving Train Test Data
            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Data Ingestion Successful')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.warning('Exception occured at Data Ingestion Stage')
            raise CustomException(e, sys)




# if __name__ == "__main__":
#     start = DataIngestion()
#     train_path, test_path = start.initiate_data_ingestion()
#     data_transform = DataTransform()
#     data_transform.initiate_data_transform(train_path, test_path)
