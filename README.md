# 🩺 Breast Cancer Classification with K-Nearest Neighbors (KNN)

Bu proje, **Scikit-Learn** kütüphanesini kullanarak **Breast Cancer Wisconsin (Diagnostic) Dataset** verileri üzerinde **K-Nearest Neighbors (KNN)** algoritması ile sınıflandırma yapmaktadır. 

## 🚀 Proje İçeriği
- **Veri Seti**: `load_breast_cancer` ile göğüs kanseri teşhisi için veri seti yüklendi.
- **Özellik ve Hedef Değerler**: Veri setindeki bağımsız değişkenler (`features`) ve bağımlı değişken (`target`) incelendi.
- **Veri Bölme**: `train_test_split` fonksiyonu ile veri seti eğitim (%80) ve test (%20) olarak ayrıldı.
- **KNN Modeli**: `KNeighborsClassifier()` modeli ile eğitim yapıldı ve test verisi üzerinde tahmin gerçekleştirildi.
- **Performans Analizi**: `confusion_matrix` ile hata matrisi oluşturuldu ve **Seaborn** kütüphanesi ile görselleştirildi.

## 📦 Kullanılan Kütüphaneler
Bu proje aşağıdaki Python kütüphanelerini kullanmaktadır:
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `pandas`

Eğer bu kütüphaneler yüklü değilse, aşağıdaki komutu kullanarak yükleyebilirsiniz:
```bash
pip install scikit-learn seaborn matplotlib pandas
```
📊 Sonuç Görselleştirme

Projede Confusion Matrix kullanılarak modelin performansı değerlendirilmektedir.

🔹 Malignant (Kötü Huylu) ve Benign (İyi Huylu) sınıfları için hata matrisi görselleştirilmiştir.

<img width="420" alt="Ekran Resmi 2025-03-13 00 47 53" src="https://github.com/user-attachments/assets/297cd2a1-cdf3-47ec-af64-e7b379946aa2" />



