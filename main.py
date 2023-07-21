import uvicorn
from fastapi import FastAPI
from os import listdir
import json
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def status():
    return {"status": "ok","message":"Working Fine"}

@app.get("/list_models")
def get_models():

    # go over models directory and list all the models
    try:
        models_list = listdir("models")

        models_list = [model.replace(".pkl", "") for model in models_list]

        return {"status": "ok", "models": models_list}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/model_details")
def model_details():

    try:

        with open("models_results.json", "r") as f:
            models_results = json.load(f)

            model_details = models_results

            return {"status":"ok","model_details":model_details}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/model_columns")
def model_columns():
    
    try:
        with open("fields.json", "r") as f:
            fields = json.load(f)

            return {"status":"ok","fields":fields}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@app.post("/predict/{model_name}")
def predict(model_name : str, data : dict):

    try:
        # load model
        loaded_model = pickle.load(open(f"models/{model_name}.pkl", "rb"))
        df = pd.DataFrame(data, index=[0])
        label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
        string_vars = ['Gender','Appointment Type', 'Weather', 'Transportation Options', 'Socio-Economic Status',
               'Reminder Sent', 'Communication Channel']


        if df["Reminder Sent"].values[0] == False:
            df["Communication Channel"] = np.nan

        print(df)

        # Label encode each categorical variable using the label encoder created above
        for col in string_vars:
            df[col] = label_encoders[col].transform(df[col])
        
        # print(df)

        prediction = loaded_model.predict(df).tolist()
        log_proba = loaded_model.predict_proba(df).tolist()


        return {"status":"ok","results":{"prediction": {
            "Will Not Show": prediction[0] == 1,
            "Will Show":prediction[0] == 0,
            "Probability of No Show": log_proba[0][1]*100,
            "Probability of Show": log_proba[0][0]*100,
        },"raw_prediction": prediction[0],
            "log_proba": log_proba}}

    except Exception as e:
        return {"status": "error", "message": str(e)}


    



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)