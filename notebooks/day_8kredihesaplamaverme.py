import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense       # type: ignore
from tensorflow.keras.optimizers import Adam    # type: ignore
from sklearn.preprocessing import StandardScaler

# --------------------
# 1) Veri
# --------------------
# giriş verileri (Yaş, Gelir) ve etiket (0/1)
X = np.array([[25,  2000],
              [30,  4000],
              [45, 10000],
              [50,  3000]], dtype=np.float32)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32)

# --------------------
# 2) Ölçekleme (z-score)
# --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # eğitim verisinden ort/ss öğren ve uygula

# --------------------
# 3) Model mimarisi
# --------------------
model = Sequential()
model.add(Dense(6, input_dim=2, activation='relu'))  # gizli katman (2 -> 6)
model.add(Dense(1, activation='sigmoid'))            # çıkış (olasılık)

# --------------------
# 4) Derleme
# --------------------
optimizer = Adam(learning_rate=0.005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# --------------------
# 5) Eğitim
# --------------------
model.fit(X_scaled, y, epochs=200, verbose=1)

# --------------------
# 6) Eğitim verisi üzerinde tahmin (kontrol)
# --------------------
tahminleme = model.predict(X_scaled, verbose=0)
print('Tahmin olasılıkları (train):\n', tahminleme)
print('Yuvarlanmış sınıf tahminleri:\n', (tahminleme > 0.5).astype(int))

# --------------------
# 7) Kullanıcıdan veri alarak canlı tahmin
# --------------------
THRESHOLD = 0.5  # karar eşiği (istersen değiştir)

while True:
    try:
        raw_age = input("Yaşınızı giriniz (çıkmak için q): ").strip()
        if raw_age.lower() == "q":
            print("Çıkıldı.")
            break

        raw_salary = input("Maaşınızı giriniz: ").strip()
        if raw_salary.lower() == "q":
            print("Çıkıldı.")
            break

        # Virgüllü girişleri de destekle
        age = float(raw_age.replace(",", "."))
        salary = float(raw_salary.replace(",", "."))

        # Şekil (1,2) olacak şekilde 2D array ve ölçekleme
        user_data = np.array([[age, salary]], dtype=np.float32)
        user_data_scaled = scaler.transform(user_data)

        # Tahmin
        proba = float(model.predict(user_data_scaled, verbose=0)[0][0])
        label = int(proba >= THRESHOLD)

        print(f"Olasılık (1 sınıfı): {proba:.4f}  ->  Tahmin sınıfı: {label}\n")

    except ValueError:
        print("Lütfen sayısal değer girin (ör. 35, 5500).")
