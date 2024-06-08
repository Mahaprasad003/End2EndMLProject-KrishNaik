#imported ilbraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#DataIngestionConfig class is basically getting the file paths and storing it
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

#DataIngestion class is responsible for ingesting the data
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    # initiate_data_ingestion is responsible for ingesting the data
    def initiate_data_ingestion(self):
        """
+        Initiates the data ingestion process.
+
+        This function reads a CSV file located at 'notebook\data\stud.csv' and performs the following steps:
+        1. Logs that the data ingestion process has started.
+        2. Reads the CSV file into a pandas DataFrame.
+        3. Logs that the dataset has been successfully read.
+        4. Creates the necessary directories for the artifacts and files.
+        5. Saves the DataFrame as a CSV file at the specified raw data path.
+        6. Logs that the artifacts and files have been created.
+        7. Performs a train-test split of the DataFrame, with a test size of 0.2 and a random state of 42.
+        8. Saves the train set as a CSV file at the specified train data path.
+        9. Saves the test set as a CSV file at the specified test data path.
+        10. Logs that the data ingestion process has completed successfully.
+        
+        Returns:
+            A tuple containing the paths to the train data and test data CSV files.
+
+        Raises:
+            CustomException: If an exception occurs during the data ingestion process.
+        """


        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Finished reading the dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Created artifacts and files')

            logging.info('Train test split initiated')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True)
            
            logging.info('Data Ingestion completed succesfully')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    