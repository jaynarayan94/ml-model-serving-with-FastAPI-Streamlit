from importlib import reload
from scripts.data_model import NLPDataInput, NLPDataOutput
from scripts import s3
import os
import torch
import pandas as pd
from transformers import pipeline
import time

from fastapi import FastAPI
from fastapi import Request
import uvicorn

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## Download ML models #####
force_download = False # True

model_name = 'tinybert-sentiment-analysis/'
local_path = 'ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)

sentiment_model = pipeline('text-classification', 
                           model=local_path, 
                           device=device)

model_name = 'tinybert-disaster-tweet/'
local_path = 'ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)

disaster_model = pipeline('text-classification', 
                           model=local_path, 
                           device=device)

## Download Ends 


@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data:NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x['label'] for x in output ]
    scores = [x['score'] for x in output ]

    output = NLPDataOutput(model_name = "tinybert-sentiment-analysis",
                           text = data.text,
                           labels = labels,
                           scores = scores,
                           prediction_time= prediction_time)

    return output

@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data:NLPDataInput):
    start = time.time()
    output = disaster_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x['label'] for x in output ]
    scores = [x['score'] for x in output ]

    output = NLPDataOutput(model_name = "tinybert-disaster-tweet",
                           text = data.text,
                           labels = labels,
                           scores = scores,
                           prediction_time= prediction_time)
    return output


if __name__=="__main__":
    uvicorn.run(app="app:app",port=8000, reload=True)