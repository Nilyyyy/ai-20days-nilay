# -*- coding: utf-8 -*-
# Çoklu Doğrusal Regresyon: [Ev_Buyuklugu, Oda_Sayisi] -> Fiyat
# Train/Test, metrikler, katsayılar + KULLANICIDAN GİRİŞLE TAHMİN

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1) Veri (örnek)
data = {
    "Ev_Buyuklugu": [120, 250, 175, 300, 220, 140, 95, 180, 260, 310],
    "Oda_Sayisi":   [  2,   4,   3,   5,   3,   2,  1,   3,   4,   5],
    "Fiyat":        [300000, 600000, 400000, 700000, 500000, 330000, 220000, 420000, 630000, 740000]
}
df = pd.DataFrame(data)

# Girdiler (X) ve hedef (y)
X = df[["Ev_Buyuklugu", "Oda_Sayisi"]]
y = df["Fiyat"]


# 2) Train/Test ayır

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Modeli eğit

model = LinearRegression()
model.fit(X_train, y_train)

# 4) Test performansı

y_pred = model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("=== Test Sonuçları ===")
print(f"MSE : {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R^2 : {r2:.4f}")

# Katsayılar (y = a1*Ev_Buyuklugu + a2*Oda_Sayisi + b)
a_boyut, a_oda = model.coef_
b = model.intercept_
print("\n=== Öğrenilen Denklem ===")
print(f"Fiyat ≈ {a_boyut:.2f} * Ev_Buyuklugu  +  {a_oda:.2f} * Oda_Sayisi  +  {b:.2f}")


# 5) KULLANICIDAN GİRİŞ AL — senin kısa versiyonla birebir

try:
    ev_buyuklugu = float(input("\nLütfen evin büyüklüğünü (m²) girin: "))
    oda_sayisi   = int(input("Lütfen oda sayısını girin: "))

    # predict 2D bekler -> [[m2, oda]]
    tahmini_fiyat = model.predict([[ev_buyuklugu, oda_sayisi]])  # ndarray -> [değer]

    print(f"\nBu evin tahmini fiyatı : {tahmini_fiyat[0]:,.0f} TL")
except ValueError:
    print("Hatalı giriş yaptın. Lütfen sayısal değer gir.")
