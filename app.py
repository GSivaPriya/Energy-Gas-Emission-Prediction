from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application=Flask(__name__)
app=application

#route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            AT=request.form.get('AT'),
            AP=request.form.get('AP'),
            AH=request.form.get('AH'),
            AFDP=request.form.get('AFDP'),
            GTEP=request.form.get('GTEP'),
            TIT=request.form.get('TIT'),
            TAT=request.form.get('TAT'),
            CDP=request.form.get('CDP')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)