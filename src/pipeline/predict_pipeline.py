import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        AT:float,
        AP: float,
        AH: float,
        AFDP: float,
        GTEP: float,
        TIT: float,
        TAT: float,
        CDP: float
        ):
        self.AT=AT
        self.AP=AP
        self.AH=AH
        self.AFDP=AFDP
        self.GTEP=GTEP
        self.TIT=TIT
        self.TAT=TAT
        self.CDP=CDP

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "AT":[self.AT],
                "AP":[self.AP],
                "AH":[self.AH],
                "AFDP":[self.AFDP],
                "GTEP":[self.GTEP],
                "TIT":[self.TIT],
                "TAT":[self.TAT],
                "CDP":[self.CDP],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)



