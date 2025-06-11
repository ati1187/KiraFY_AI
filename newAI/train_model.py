from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from a import temizle_veri  # Veri temizleme fonksiyonunu import et

import pickle
import os

# ✅ Temiz veriyi yükle
temiz_df = temizle_veri("HouseData.csv")

# 🎯 Hedef ve öznitelikleri ayır
y = temiz_df['price']
X = temiz_df.drop(['price'], axis=1)

# 📊 Eğitim/test bölünmesi
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Model tanımı
model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(x_train, y_train)

# 🧪 Skorlar
y_pred = model.predict(x_test)
print("📊 Model performansı:")
print("🔹 R² Skoru:", r2_score(y_test, y_pred))
print("🔹 MAE (Ortalama Mutlak Hata):", mean_absolute_error(y_test, y_pred))

# 💾 Modeli kaydetmek için klasör oluştur
os.makedirs('model', exist_ok=True)

# 📦 Model + kolon bilgisi birlikte kaydedilir
with open('model/house_price_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'columns': X.columns.tolist()}, f)

print("✅ Model başarıyla 'model/house_price_model.pkl' dosyasına kaydedildi.")
