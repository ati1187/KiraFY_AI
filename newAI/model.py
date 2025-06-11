import pickle
import pandas as pd
import os

class HousePriceModel:
    def __init__(self, model_path='model/house_price_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.columns = model_data['columns']
        print("✅ Model başarıyla yüklendi.")

    def predict(self, input_data: dict) -> float:
        """
        input_data: {"district": "Kadıköy", "NumberOfRooms": "3+1", ...}
        """
        # DataFrame'e çevir (tek satır)
        df = pd.DataFrame([input_data])

        # NetSquareMeters temizle
        if 'NetSquareMeters' in df.columns:
            df['NetSquareMeters'] = df['NetSquareMeters'].astype(str).str.replace(' m2', '', regex=False)
            df['NetSquareMeters'] = pd.to_numeric(df['NetSquareMeters'], errors='coerce')

        # BuildingAge ordinal map
        age_map = {
            "0": 0, "1-5": 1, "5-10": 2, "11-15": 3, "16-20": 4, "21 Ve Üzeri": 5
        }
        df['BuildingAge'] = df['BuildingAge'].map(age_map)

        # NumberOfRooms parse et
        df['NumberOfRooms'] = df['NumberOfRooms'].str.extract(r'(\d+)').astype(float)

        # One-hot encoding
        df = pd.get_dummies(df)

        # Eksik sütunları tamamla
        df = df.reindex(columns=self.columns, fill_value=0)

        # Tahmin
        prediction = self.model.predict(df)[0]
        return round(float(prediction), 2)
