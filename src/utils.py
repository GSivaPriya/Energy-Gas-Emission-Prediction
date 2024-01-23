import os
import sys
import numpy as np 
import pandas as pd 
from src.exception import CustomException
import dill
from sklearn.metrics import (r2_score,mean_squared_error)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for model_name,model in models.items():
            para = params[model_name]

            # Wrap the model with MultiOutputRegressor
            model = MultiOutputRegressor(model)

            gs = GridSearchCV(model, para, cv=4)
            gs.fit(X_train, y_train)

            # Set the best parameters to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            train_model_mse=mean_squared_error(y_train,y_train_pred)
            test_model_mse=mean_squared_error(y_test,y_test_pred)

            train_model_rmse=np.sqrt(train_model_mse)
            test_model_rmse=np.sqrt(test_model_mse)

            report[model_name]={
                'train_r2_score':train_model_score,
                'test_r2_score':test_model_score,
                'train_rmse':train_model_rmse,
                'test_rmse':test_model_rmse
            }
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)  