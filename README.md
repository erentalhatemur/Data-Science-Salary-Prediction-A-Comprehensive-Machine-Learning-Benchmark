# Data-Science-Salary-Prediction-A-Comprehensive-Machine-Learning-Benchmark
# Data Science Salary Prediction: A Comprehensive Machine Learning Benchmark

Bu proje, veri bilimi sektöründeki maaşları etkileyen faktörleri analiz etmek ve farklı makine öğrenmesi algoritmaları kullanarak maaş tahmini yapmak amacıyla geliştirilmiştir. Proje, veri ön işlemeden model ayarlamaya (Hyperparameter Tuning) ve modellerin performans karşılaştırmasına kadar tüm veri bilimi yaşam döngüsünü kapsamaktadır.

## 🧠 Proje Özeti ve Teorik Altyapı
[cite_start]Proje, doğrusal olmayan regresyon problemlerine odaklanarak beynin işleyişinden ilham alan **Yapay Sinir Ağları (ANN)** gibi ileri düzey modelleri ve ağaç tabanlı algoritmaları içermektedir[cite: 7, 8].

* [cite_start]**Nöron Modeli:** Yapay nöronlar, girdileri ağırlıklandırıp toplayan (Aggregation Function) ve bir eşiği aşınca çıktı üreten (Activation Function) matematiksel soyutlamalardır[cite: 10, 86].
* [cite_start]**Öğrenme Süreci:** Modellerin eğitilmesi, hata fonksiyonunu minimize edecek en uygun ağırlıkların (Weights) optimizasyon algoritmasıyla bulunması sürecidir[cite: 180, 181, 182].

## 🛠️ Teknik İşlemler
* [cite_start]**Veri Ön İşleme:** `StandardScaler` kullanılarak hem bağımsız değişkenler hem de hedef değişken (Salary) normalize edilmiştir[cite: 248, 258].
* [cite_start]**Kategorik Dönüşüm:** `pd.get_dummies` yöntemiyle nominal değişkenler sayısal forma getirilmiştir[cite: 235].
* [cite_start]**Model Optimizasyonu:** `GridSearchCV` kullanılarak `alpha`, `hidden_layer_sizes` ve `depth` gibi hiperparametreler optimize edilmiştir[cite: 428, 435].

## 📊 Karşılaştırmalı Performans Sonuçları
Tüm modeller **Train R2** (Eğitim başarısı) ve **Test R2** (Genelleme yeteneği) skorlarına göre değerlendirilmiştir:

| Model | Train $R^2$ | Test $R^2$ | Durum Analizi |
| :--- | :--- | :--- | :--- |
| **CART (Tuned)** | **0.6595** | **0.5010** | **En Başarılı Model (Şampiyon)** |
| **Lasso** | 0.0540 | 0.4876 | Dengeli Tahminleme |
| **Random Forest** | 0.6341 | 0.4470 | Yüksek Öğrenme Kapasitesi |
| **ANN (Tuned)** | **0.7798** | **0.2724** | **Aşırı Öğrenme (Overfitting)** |



## 💡 Temel Bulgular ve Yorumlar
1. **Şampiyon Model:** %50.10 test başarısıyla **Tuned CART** modelidir. Maaş verilerindeki hiyerarşik "eğer-ise" kurallarını en iyi Karar Ağacı yapısı yakalamıştır.
2. **Değişken Önemliliği:** Yapılan analizler sonucunda maaş üzerindeki en belirleyici faktörlerin **Kıdem Seviyesi (Experience Level)** ve **Lokasyon (US/City)** olduğu saptanmıştır.
3. **Overfitting Uyarısı:** Yapay Sinir Ağları (ANN), eğitim setinde %77.98 başarı gösterse de test setinde düşük kalarak bu veri boyutu için aşırı ezberleme (overfitting) eğilimi sergilemiştir.



## 📋 Gereksinimler
* Python 3.x
* Pandas, NumPy
* Scikit-Learn
* Matplotlib, Seaborn

---
**Geliştiren:** Eren Talha Temur
