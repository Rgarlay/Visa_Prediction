from src.logger import logging
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object,evaluate_model,remove_outliers

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier  # Requires XGBoost library
from sklearn.metrics import recall_score



@dataclass
class model_trainer_path():
    trained_model_file_path = os.path.join('archieve','trainer.pkl')

class DataTrainer():
    def __init__(self):
        self.model_trainer_caller=model_trainer_path()
        

    def initiate_model_training(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test =(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Data split into training and testing sets.")

            models = {
                        "Random Forest": RandomForestClassifier(),
                        "Support Vector Classifier": SVC(),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "Logistic Regression": LogisticRegression(),
                        "AdaBoost": AdaBoostClassifier(),
                        "XGBoost": XGBClassifier(),
                    }
            
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Support Vector Classifier": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.05,],
                    "max_depth": [3, 4, 5],
                    "subsample": [0.7, 0.8, 1.0]
                },
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs"],
                    "penalty": ["l2", "none"],
                    "max_iter": [100, 200,50]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1,0.5],
                    "algorithm": ["SAMME", "SAMME.R"]
                },
                "XGBoost": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.05],
#                    "max_depth": [3, 4, 5],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                }
            }

            logging.info("Evaluating models using RandomizedSearchCV...")
            
            model_result:dict = evaluate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test
                                           ,models=models,params=params)
            
            logging.info(f"Model results: {model_result}")

            best_model_value = max(model_result.values())
            logging.info(f"Best model value: {best_model_value}")
            
            ##Converted the models values into list format, took the index of max one, then converted the model key into list
            ##and used the corrosponding index to find the model name.
            
            best_model_name = list(model_result.keys())[list(model_result.values()).index(best_model_value)]
            logging.info(f"Best model found: {best_model_name}")


            best_model = models[best_model_name]

            print(f'{best_model_name} is the best model')


            if best_model_value<0.6:
                logging.warning("No best model found with recall greater than 0.6")
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_caller.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved at {self.model_trainer_caller.trained_model_file_path}")


            predictor = best_model.predict(X_test)
            logging.info("Predictions made on test set.")


            recall_val = recall_score(y_test,predictor,pos_label='Certified')
            logging.info(f"Final recall score on test set: {recall_val}")

            logging.info("Traning and entire process is complete.")
            
            return recall_val

        except Exception as e:
         logging.error(f"Error in model training: {e}")
         raise CustomException(e,sys)
