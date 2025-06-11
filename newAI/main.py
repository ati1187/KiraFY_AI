from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from model import HousePriceModel
from pydantic import BaseModel

app = FastAPI()

# 🔧 Statik HTML klasörü
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🏠 Ana sayfa: HTML form
@app.get("/")
def get_form():
    return FileResponse("static/index.html")

# ✅ Modeli yükle
model = HousePriceModel()

# 📥 Formdan gelecek veri şeması
class PredictionInput(BaseModel):
    district: str
    GrossSquareMeters: float
    NetSquareMeters: str
    NumberOfRooms: str
    FloorLocation: str
    HeatingType: str
    BuildingAge: str
    NumberFloorsofBuilding: int
    NumberOfBathrooms: int
    NumberOfWCs: int
    InsideTheSite: str
    MortgageStatus: str
    Swap: str
    Balcony: str
    PriceStatus: str
    CreditEligibility: str
    ItemStatus: str
    BuildStatus: str

# 🔮 Tahmin endpoint’i
@app.post("/predict")
def predict_price(input: PredictionInput):
    try:
        input_dict = input.dict()
        prediction = model.predict(input_dict)
        return {"tahmin_edilen_fiyat": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
