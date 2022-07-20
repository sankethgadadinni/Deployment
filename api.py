from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List

from fastapi.encoders import jsonable_encoder


import pycaret
from pycaret.regression import *
import uvicorn





app = FastAPI()


class MLItem(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

loaded_model = load_model('insurance')


@app.post('/predict')
async def predict_charges(user_input: MLItem):

    json_user_input=jsonable_encoder(user_input)

    data_in = pd.DataFrame([[json_user_input['age'], json_user_input['sex'], json_user_input['bmi'], json_user_input['children'], json_user_input['smoker'], json_user_input['region']]])
    
    data_in.columns = ['age','sex','bmi','children','smoker','region']

    prediction = predict_model(loaded_model, data = data_in)

    chrages = list(prediction['Label'])

    return {"Charges" : chrages}



