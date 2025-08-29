import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_excel("data/karar_agaci_veri_100.xlsx")

X = df[['Yas', 'Kan_Basinci', 'Kolesterol']]  
y = df['Hastalik']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)



yas = int(input("Yaşınızı girin: "))
kan_basinci = int(input("Kan basıncınızı girin: "))
kolesterol = int(input("Kolesterol seviyenizi girin: "))   

kullanici_verisi = pd.DataFrame([[yas, kan_basinci, kolesterol]], columns=['Yas', 'Kan_Basinci', 'Kolesterol'])

tahmin = classifier.predict(kullanici_verisi)
sonuc = "Hastalık var" if tahmin[0] == 1 else "Hastalık Yok"
print(f"Tahmin : {sonuc}")
