import os
import sys
import numpy as np
import dill
from .exception import CustomException
from sklearn.metrics import recall_score,f1_score,accuracy_score
from .logger import logging
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
            raise CustomException(e,sys)
    
def load_object(file_path):

    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
                
    except Exception as e:
        raise CustomException(e,sys)

def remove_outliers(df):

    """Remove outliers using IQR method."""
    cols = df.select_dtypes(include=[np.number]).columns  # Apply to all numeric columns
    df_filtered = df.copy()  # Create a copy to avoid modifying the original DataFrame

    for col in cols:
        q1 = df_filtered[col].quantile(0.25)
        q3 = df_filtered[col].quantile(0.75)
        IQR = q3 - q1
        upper_bound = q3 + 1.5 * IQR
        lower_bound = q1 - 1.5 * IQR

        # Filter the DataFrame for the current column
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]  # Changed line
    
    return df_filtered


def evaluate_model(X_train, X_test, y_train, y_test, models, params):   
    try:
        logging.info("Starting model evaluation...")
        recall_test_final = {}
        label_encoder = None
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            parameters = params[model_name]
            
            if model_name == "XGBoost":
                logging.info("Applying label encoding for XGBoost.")
                label_encoder = LabelEncoder()

                y_train_encoded = label_encoder.fit_transform(y_train)
            else:
                y_train_encoded = y_train

            logging.info(f"Training model: {model_name} with RandomizedSearchCV")
            randn_search = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=3, n_jobs=-1, refit=False)
            randn_search.fit(X_train, y_train_encoded)

            logging.info(f"Best parameters found for {model_name}: {randn_search.best_params_}")
            model.set_params(**randn_search.best_params_)
            logging.info(f"Training model: {model_name}")

            model.fit(X_train, y_train_encoded)
            logging.info(f"Model {model_name} fitted with training data")

            y_pred = model.predict(X_test)

            if label_encoder:
                y_pred = label_encoder.inverse_transform(y_pred)

            logging.info(f"Model {model_name} trained. Making predictions...")

            test_recall = recall_score(y_test, y_pred, pos_label='Certified')
            logging.info(f"Recall score for {model_name}: {test_recall}")

            recall_test_final[model_name] = test_recall

        logging.info("Model evaluation completed successfully.")
        return recall_test_final
     
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)
            