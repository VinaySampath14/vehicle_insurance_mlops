import sys
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.pipline.prediction_pipeline import PredictionPipeline, VehicleData
from src.pipline.training_pipeline import TrainPipeline
from src.exception import MyException
from src.logger import logging

app = FastAPI(title="Vehicle Insurance Cross-Sell Predictor")
templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    gender: str
    age: int
    driving_license: int
    region_code: float
    previously_insured: int
    vehicle_age: str
    vehicle_damage: str
    annual_premium: float
    policy_sales_channel: float
    vintage: int


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(data: PredictRequest):
    try:
        vehicle_data = VehicleData(
            gender=data.gender,
            age=data.age,
            driving_license=data.driving_license,
            region_code=data.region_code,
            previously_insured=data.previously_insured,
            vehicle_age=data.vehicle_age,
            vehicle_damage=data.vehicle_damage,
            annual_premium=data.annual_premium,
            policy_sales_channel=data.policy_sales_channel,
            vintage=data.vintage,
        )
        df = vehicle_data.get_data_as_dataframe()
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(df)
        return JSONResponse({"prediction": int(prediction[0])})

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise MyException(e, sys)


@app.get("/train")
async def train():
    try:
        logging.info("Training triggered via /train endpoint.")
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return JSONResponse({"status": "Training completed successfully."})
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise MyException(e, sys)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
