import sys
import pandas as pd
from ..exception import CustomException
from ..utils import load_object



class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
         try:
            model_path = "archieve/trainer.pkl"
            preprocessor_path = "archieve/preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
         except Exception as e:
             raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                continent: str,
                education_of_employee: str,
                has_job_experience: str,
                requires_job_training: str,
                no_of_employees: int,
                yr_of_estab: int,
                region_of_employment: str,
                prevailing_wage: int,
                unit_of_wage: str,
                full_time_position: str
                 ):
      
          
            self.continent = continent

            self.education_of_employee = education_of_employee

            self.has_job_experience = has_job_experience

            self.requires_job_training = requires_job_training

            self.no_of_employees = no_of_employees

            self.yr_of_estab = yr_of_estab

            self.region_of_employment = region_of_employment

            self.prevailing_wage = prevailing_wage

            self.unit_of_wage = unit_of_wage

            self.full_time_position = full_time_position

        
    def get_data_into_data_frame(self):
          
          try:
              custom_data_input_dict = {
                  'continent' : [self.continent],
                  'education_of_employee' : [self.education_of_employee],
                  'has_job_experience' : [self.has_job_experience],
                  'requires_job_training' : [self.requires_job_training],
                  'no_of_employees' : [self.no_of_employees],
                  'yr_of_estab' : [self.yr_of_estab],
                  'region_of_employment' : [self.region_of_employment],
                  'prevailing_wage' : [self.prevailing_wage],
                  'unit_of_wage' : [self.unit_of_wage],
                  'full_time_position' : [self.full_time_position],                
              }
              return pd.DataFrame(custom_data_input_dict)
          except Exception as e:
              raise CustomException(e,sys)
                
