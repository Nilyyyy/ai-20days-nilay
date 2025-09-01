


import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#veriyi yükle
df = pd.read_excel(
    r"C:\Users\bunco\OneDrive\Masaüstü\20days-Nilay\data\veri_on_isleme_ve_ozellik_muhendisligi.xlsx"
)
#cinseyet ve gelirleri ortalama ile doldurmak
df.fillna(df['Gelir'].mean(), inplace=True)
#cinsiyet ve Meslek Sütunlarını sayısal hale getirelim
le = LabelEncoder()
df['Cinsiyet'] = le.fit_transform(df['Cinsiyet'])
df['Meslek'] = le.fit_transform(df['Meslek'])
#giriş ve çıktı verilerini girelim
X = df[['Yaş','Meslek','Cinsiyet']] # girdi
y = df['Gelir'] # çıktı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# #modeli oluştur ve eğit
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
#modeli test
linear_accuracy = linear_model.score(X_test,y_test)
print(f'Modelin Doğruluk oranı : {linear_accuracy * 100:.2f}%')
#daha karmaşık bir model kullan

rf_model = RandomForestRegressor(n_estimators=80 , random_state=1)
rf_model.fit(X_train,y_train)
rf_accuracy = rf_model.score(X_test, y_test)

print(f'Forest Modelin Doğruluk oranı : {rf_accuracy* 100:.2f}%')
# Kullanıcıdan veri alalım
print("Lütfen tahmin için aşağıdaki bilgileri giriniz:")
yas = int(input("Yaş: "))
meslek = input("Meslek (Mühendis, Doktor, Öğretmen, Avukat): ")
cinsiyet = input("Cinsiyet (Erkek, Kadın): ")
#kullanıcıdan alınan mesleği kodlyalım (Label Encoding)

if cinsiyet == 'Erkek':
    cinsiyet_kod = 0
elif cinsiyet == 'Kadın':
    cinsiyet_kod = 1
else:
    raise ValueError('Geçersiz cinsiyet değeri')

meslek_kod = le.transform([meslek])[0]
yeni_veri = pd.DataFrame([[yas,meslek_kod,cinsiyet_kod]] , columns=['Yaş','Meslek','Cinsiyet'])
yeni_veri_scaled = scaler.transform(yeni_veri)
tahmin = rf_model.predict(yeni_veri_scaled)
print(f'Tahmini Ortalama Maaş : {tahmin[0]:.2f} TL')