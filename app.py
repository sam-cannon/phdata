from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List
import logging
import io

app = FastAPI()

# Load your trained machine learning model
model = pickle.load(open('/Users/charlenehack/Desktop/Sam/fastapi_docker_ml/gradient_boosted_model.pkl', 'rb'))

class Item(BaseModel):
    file: UploadFile

@app.post("/predict/")
async def predict(file: UploadFile):

    try:
        #read in zipcode demo file
        demo = pd.read_csv('zipcode_demographics.csv')

        #read in features needed for generating predictions from the model
        features = pd.read_json('model_features.json')[0].tolist()

        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        #merge our uploaded df into our demographic csv 
        df = df.merge(demo, on = 'zipcode')

        #order columns in df the same way as the data that was used to train the model
        df = df[features]

        # Make predictions using the loaded model
        predictions = model.predict(df)

    #if error occurs, log and return the error to the command line
    except Exception as e:
        
        logging.error(f"Error in prediction: {str(e)}")
        
        return {"error": "Failed to make predictions."}

    # Convert predictions to a JSON response
    response = {"predictions": predictions.tolist()}
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
