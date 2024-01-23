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


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models



@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and testing inout data")
            X_train = train_arr[:, :-3]  
            y_train = train_arr[:, -3:]   
            
            X_test = test_arr[:, :-3]    
            y_test = test_arr[:, -3:]

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                 #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'estimator__splitter':['best','random'],
                    'estimator__max_features':[2, 4, 6, 8],
                    },
                "Random Forest":{
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],      
                    'estimator__max_features':[2, 4, 6, 8],
                    'estimator__n_estimators': [2,4,8]
                },
                "Gradient Boosting":{
                    #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'estimator__learning_rate':[.1,.01,.05],
                     #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #'estimator__criterion':['squared_error', 'friedman_mse'],
                    #'estimator__max_features':['auto','sqrt','log2'],
                    'estimator__n_estimators': [2,4,8]
                     },
                 "Linear Regression": {},
                "K-Neighbors Regressor": {
                'estimator__n_neighbors': [3,5],  # Number of neighbors to consider
                #'weights': ['uniform', 'distance'],  # Weight function used in prediction
                #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
                'estimator__leaf_size': [5,10,20],  # Leaf size passed to BallTree or KDTree
                #estimator__p': [1, 2],  # Power parameter for the Minkowski metric
                'estimator__n_jobs': [-1]  # If using scikit-learn, this parameter can be used to parallelize the computation
                        },
                "AdaBoost Regressor":{
                    'estimator__learning_rate':[.1,.01,0.05],
                     #'estimator__loss':['linear','square','exponential'],
                    'estimator__n_estimators': [2,4,8]
                        }
                
                    }   


            model_report: dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            best_model_name = max(model_report, key=lambda x: model_report[x]['test_r2_score'])
            best_model = models[best_model_name]

            best_test_r2_score = model_report[best_model_name]['test_r2_score']
            

            if best_test_r2_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name}")
            logging.info(f"Best model R2 score: {best_test_r2_score}")
            

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            best_model.fit(X_train,y_train)
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
