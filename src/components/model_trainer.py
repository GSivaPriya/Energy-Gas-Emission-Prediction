import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor

)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

#for every component create a config file

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("Splitting training and testing inout data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-3],
                train_array[:,-3],
                test_array[:,:-3],
                test_array[:,-3]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                #"XGBRegressor":XGBRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            model_report: dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            '''best_model_score= max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")'''

            best_model_name = max(model_report, key=lambda x: model_report[x]['test_r2_score'])
            best_model = models[best_model_name]

            best_test_r2_score = model_report[best_model_name]['test_r2_score']
            best_test_rmse = model_report[best_model_name]['test_rmse']

            if best_test_r2_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name}")
            logging.info(f"Best model test R2 score: {best_test_r2_score}")
            logging.info(f"Best model test RMSE: {best_test_rmse}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
