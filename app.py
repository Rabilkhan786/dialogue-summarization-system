from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from fastapi.responses import Response
import uvicorn
import os

from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI()


# LOAD MODEL ONLY ONCE
obj = PredictionPipeline()


# REQUEST SCHEMA
class TextInput(BaseModel):
    text: str



@app.get("/")
@app.head("/")  
async def index():
    return {"status": "healthy", "message": "Dialogue Summarizer API is running"}

@app.get("/train")
async def training():

    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(data: TextInput):

    try:

        summary = obj.predict(data.text)

        return {
            "summary": summary
        }

    except Exception as e:
        return {
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)