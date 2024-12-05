import os
import sys
import pandas as pd
from dataclasses import dataclass
from ..exception import CustomException
from ..logger import logging
from sklearn.model_selection import train_test_split
from .data_transformation import DataTransformation
from .model_tranier import DataTrainer

@dataclass
class Data_ingestion():
    def __init__(self):
        self.train_data_path = os.path.join('archieve','train.csv')
        self.test_data_path = os.path.join('archieve','test.csv')
        self.raw_data_path = os.path.join('archieve','raw.csv')
    
class initiate_data_ingestion():
    def __init__(self):
        
        self.initiate_data_ingest = Data_ingestion()
    
    def code_initiating(self):
        try:
            logging.info("Data Ingestion has been initiated")

            df1 = pd.read_csv(r'C:\Users\rgarlay\Desktop\DS\VISA_APPROVAL\Visa_Prediction\archive\Visa_Predection_Dataset.csv') 
            
            logging.info("Data has been imported successfully")

            columns_to_drop = 'case_id'

            df = df1.drop(columns=[columns_to_drop],axis=1)

            df.to_csv(self.initiate_data_ingest.raw_data_path, index=False, header=True)

            logging.info("Data has been stoed in raw file successfully")

            os.makedirs(os.path.dirname(self.initiate_data_ingest.train_data_path), exist_ok=True)

            logging.info("the directory has been created and verified for exostance.")

            train_data,test_data = train_test_split(df, test_size=0.2, random_state=42)

            logging.info('Data has been split successfully')

            train_data.to_csv(self.initiate_data_ingest.train_data_path, header=True,index=False)

            test_data.to_csv(self.initiate_data_ingest.test_data_path, header=True,index=False)

            logging.info('Split data has been stored successfully')

            logging.info('Ingestion is complete')

            return self.initiate_data_ingest.train_data_path,self.initiate_data_ingest.test_data_path
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = initiate_data_ingestion()
    train_data, test_data = obj.code_initiating()

    data_transoformation  = DataTransformation()
    train_arr,test_arr,addditional_info = data_transoformation.data_transform_initiate(train_data,test_data)

    data_trainer = DataTrainer()
    print(f"The recall value is: {data_trainer.initiate_model_training(train_arr,test_arr)}")
    
