from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import numpy as np
from joblib import load

# Load the trained model
model = load('./cov_type_model_v1.joblib')

# Initialize FastAPI app
app = FastAPI()

# Define Enums for dropdown lists
class WildernessArea(str, Enum):
    area1 = "area1"
    area2 = "area2"
    area3 = "area3"
    area4 = "area4"

class SoilType(str, Enum):
    type1 = "type1"
    type2 = "type2"
    type3 = "type3"
    type4 = "type4"
    type5 = "type5"
    type6 = "type6"
    type7 = "type7"
    type8 = "type8"
    type9 = "type9"
    type10 = "type10"
    type11 = "type11"
    type12 = "type12"
    type13 = "type13"
    type14 = "type14"
    type15 = "type15"
    type16 = "type16"
    type17 = "type17"
    type18 = "type18"
    type19 = "type19"
    type20 = "type20"
    type21 = "type21"
    type22 = "type22"
    type23 = "type23"
    type24 = "type24"
    type25 = "type25"
    type26 = "type26"
    type27 = "type27"
    type28 = "type28"
    type29 = "type29"
    type30 = "type30"
    type31 = "type31"
    type32 = "type32"
    type33 = "type33"
    type34 = "type34"
    type35 = "type35"
    type36 = "type36"
    type37 = "type37"
    type38 = "type38"
    type39 = "type39"
    type40 = "type40"

# Define the input data model
class CovTypeInput(BaseModel):
    elevation: float
    aspect: float
    slope: float
    horizontal_distance_to_hydrology: float
    vertical_distance_to_hydrology: float
    horizontal_distance_to_roadways: float
    hillshade_9am: float
    hillshade_noon: float
    hillshade_3pm: float
    horizontal_distance_to_fire_points: float
    wilderness_area: WildernessArea
    soil_type: SoilType

# Define the prediction endpoint
@app.post('/predict')
def predict(input_data: CovTypeInput):
    wilderness_area_mapping = {
        WildernessArea.area1: [1, 0, 0, 0],
        WildernessArea.area2: [0, 1, 0, 0],
        WildernessArea.area3: [0, 0, 1, 0],
        WildernessArea.area4: [0, 0, 0, 1]
    }
    
    soil_type_mapping = {
        SoilType.type1: [1] + [0] * 39,
        SoilType.type2: [0, 1] + [0] * 38,
        SoilType.type3: [0] * 2 + [1] + [0] * 37,
        SoilType.type4: [0] * 3 + [1] + [0] * 36,
        SoilType.type5: [0] * 4 + [1] + [0] * 35,
        SoilType.type6: [0] * 5 + [1] + [0] * 34,
        SoilType.type7: [0] * 6 + [1] + [0] * 33,
        SoilType.type8: [0] * 7 + [1] + [0] * 32,
        SoilType.type9: [0] * 8 + [1] + [0] * 31,
        SoilType.type10: [0] * 9 + [1] + [0] * 30,
        SoilType.type11: [0] * 10 + [1] + [0] * 29,
        SoilType.type12: [0] * 11 + [1] + [0] * 28,
        SoilType.type13: [0] * 12 + [1] + [0] * 27,
        SoilType.type14: [0] * 13 + [1] + [0] * 26,
        SoilType.type15: [0] * 14 + [1] + [0] * 25,
        SoilType.type16: [0] * 15 + [1] + [0] * 24,
        SoilType.type17: [0] * 16 + [1] + [0] * 23,
        SoilType.type18: [0] * 17 + [1] + [0] * 22,
        SoilType.type19: [0] * 18 + [1] + [0] * 21,
        SoilType.type20: [0] * 19 + [1] + [0] * 20,
        SoilType.type21: [0] * 20 + [1] + [0] * 19,
        SoilType.type22: [0] * 21 + [1] + [0] * 18,
        SoilType.type23: [0] * 22 + [1] + [0] * 17,
        SoilType.type24: [0] * 23 + [1] + [0] * 16,
        SoilType.type25: [0] * 24 + [1] + [0] * 15,
        SoilType.type26: [0] * 25 + [1] + [0] * 14,
        SoilType.type27: [0] * 26 + [1] + [0] * 13,
        SoilType.type28: [0] * 27 + [1] + [0] * 12,
        SoilType.type29: [0] * 28 + [1] + [0] * 11,
        SoilType.type30: [0] * 29 + [1] + [0] * 10,
        SoilType.type31: [0] * 30 + [1] + [0] * 9,
        SoilType.type32: [0] * 31 + [1] + [0] * 8,
        SoilType.type33: [0] * 32 + [1] + [0] * 7,
        SoilType.type34: [0] * 33 + [1] + [0] * 6,
        SoilType.type35: [0] * 34 + [1] + [0] * 5,
        SoilType.type36: [0] * 35 + [1] + [0] * 4,
        SoilType.type37: [0] * 36 + [1] + [0] * 3,
        SoilType.type38: [0] * 37 + [1] + [0] * 2,
        SoilType.type39: [0] * 38 + [1] + [0] * 1,
        SoilType.type40: [0] * 39 + [1]
    }
    
    data = [
        input_data.elevation,
        input_data.aspect,
        input_data.slope,
        input_data.horizontal_distance_to_hydrology,
        input_data.vertical_distance_to_hydrology,
        input_data.horizontal_distance_to_roadways,
        input_data.hillshade_9am,
        input_data.hillshade_noon,
        input_data.hillshade_3pm,
        input_data.horizontal_distance_to_fire_points,
        *wilderness_area_mapping[input_data.wilderness_area],
        *soil_type_mapping[input_data.soil_type]
    ]
    
    data_array = np.array(data).reshape(1, -1)
    prediction = model.predict(data_array)
    return {"prediction": int(prediction[0])}

# To run the app, use the following command:
# uvicorn main:app --reload

# Build the docker image
# docker build -t fastapi-covtype

# Run the docker container
# docker run -d -p 8000:80 fastapi-covtype


