# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer

# 1-Veri setini yüklüyoruz
breast = load_breast_cancer()

# 2) Veri setiyle ilgili bilgileri yazdırıyoruz (özellik adları, hedef adları, target, data)
print(breast.feature_names)
print(breast.target_names)
print(breast.target)
print(breast.data)

# 3-X ve Y değişkenlerini belirliyoruz
X = breast.data
Y = breast.target

# 4-Eğitim ve test olarak ayırma (train_test_split)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.20, 
                                                    random_state=0)
print("Eğitim veri seti boyutu=", len(X_train))
print("Test veri seti boyutu=", len(X_test))

# 5-KNeighborsClassifier modelini kuruyoruz
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)

# 6-Test verisiyle tahmin yapıyoruz
Y_tahmin = model.predict(X_test)

# 7-Confusion Matrix (hata matrisi) oluşturuyoruz
from sklearn.metrics import confusion_matrix
hata_matrisi = confusion_matrix(Y_test, Y_tahmin)
print(hata_matrisi)

# 8-Matplotlib ve Seaborn ile görselleştiriyoruz
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


index = ['malignant','benign']
columns = ['malignant','benign']

hata_goster = pd.DataFrame(hata_matrisi, columns, index)
plt.figure(figsize=(10,6))
sns.heatmap(hata_goster, annot=True)
plt.show()
