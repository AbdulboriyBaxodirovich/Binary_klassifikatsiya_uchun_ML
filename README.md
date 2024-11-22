### README: Chekishni Bashoratlash Loyihasi (Smoking Prediction Project)  

Ushbu loyiha tibbiy va demografik ma'lumotlardan foydalanib, chekish odatlarini bashorat qilishga qaratilgan. Quyida loyihaning jarayoni, ishlatilgan metodlar va optimizatsiya hamda baholash usullari haqida batafsil ma'lumot berilgan.  

---

### **Dataset haqida (Dataset Overview)**  
Datasetda quyidagi ustunlar mavjud:  
- **id**: Har bir inson uchun noyob identifikator (Unique identifier for each individual)  
- **age**: Yosh, yillarda (Age in years)  
- **height(cm)**: Bo‘yi, santimetrda (Height in centimeters)  
- **weight(kg)**: Vazni, kilogrammda (Weight in kilograms)  
- **waist(cm)**: Bel atrofi, santimetrda (Waist circumference in centimeters)  
- **eyesight(left/right)**: Chap va o‘ng ko‘zning ko‘rish qobiliyati (Vision capability for the left and right eyes)  
- **hearing(left/right)**: Chap va o‘ng quloqning eshitish qobiliyati (Hearing capability for the left and right ears)  
- **systolic/relaxation**: Qon bosimi ko‘rsatkichlari (Blood pressure readings)  
- **fasting blood sugar**: Qondagi shakar miqdori (mg/dL) (Blood sugar levels (mg/dL))  
- **Cholesterol, triglyceride, HDL, LDL**: Qon lipid profili ko‘rsatkichlari (Lipid profile parameters)  
- **hemoglobin**: Qondagi gemoglobin darajasi (Hemoglobin levels in the blood)  
- **Urine protein**: Siydikdagi oqsil miqdori (Protein levels in urine)  
- **serum creatinine**: Qondagi kreatinin darajasi (Creatinine levels in the blood)  
- **AST, ALT, GTP**: Jigar fermentlari ko‘rsatkichlari (Liver enzyme readings)  
- **dental caries**: Tish holati indikatori (Dental condition indicator)  
- **smoking**: Maqsadli ustun: chekish odati (1 = chekadi, 0 = chekmaydi) (Target variable: 1 = smoker, 0 = non-smoker)  

---

### **Ishlatilgan vositalar va kutubxonalar (Tools and Libraries)**  
- **Python**: Dasturlash tili (Programming language for implementation)  
- **Pandas & NumPy**: Ma’lumotlarni qayta ishlash va matematik hisoblar uchun (Data manipulation and numerical computations)  
- **Scikit-learn**: Mashinani o‘rganish modellarini yaratish, metrikalar va kross-validatsiya (Machine learning models, metrics, and cross-validation)  
- **Optuna**: Gipermetrlarni optimallashtirish kutubxonasi (Hyperparameter tuning library)  

---

### **Loyiha jarayoni (Project Workflow)**  

#### 1. **Ma'lumotlarni ajratish (Data Splitting)**  
Ma'lumotlar `train_test_split` orqali o‘quv (train) va test to‘plamlariga bo‘lindi.  
```python
X = df.drop('smoking', axis=1)
y = df['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 2. **Optuna yordamida gipermetrlarni sozlash (Hyperparameter Tuning with Optuna)**  
RandomForest modelining gipermetrlari (masalan, `n_estimators`, `max_depth`) Optuna yordamida optimallashtirildi.  
```python
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    ...
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    score = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score
```

#### 3. **Eng yaxshi parametrlar bilan modelni o‘rgatish (Model Training with Best Parameters)**  
Optuna topgan eng yaxshi parametrlar asosida RandomForestClassifier va DecisionTreeClassifier yaratildi. Ushbu modellar `VotingClassifier` yordamida birlashtirildi.  
```python
rf_model = RandomForestClassifier(**best_params, random_state=42)
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('dt', dt_model)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
```

#### 4. **Modellarni baholash (Model Evaluation)**  
ROC AUC va o‘quv egri chiziqlari yordamida modellar baholandi.  
```python
auc_score = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
```

#### 5. **Kalibratsiya va Brier Score (Calibration and Brier Score)**  
Random Forest modeli `CalibratedClassifierCV` yordamida kalibrlashdan o‘tkazildi va Brier Score kamaygani kuzatildi.  
```python
calibrated_rf = CalibratedClassifierCV(estimator=rf_model, method='sigmoid', cv=5)
calibrated_rf.fit(X_train, y_train)
brier_calibrated = brier_score_loss(y_test, y_prob_calibrated)
```

---

### **Natijalar (Results)**  
- **Gipermetr optimallashtirish (Hyperparameter Optimization)**: Optuna yordamida modelning eng yaxshi parametrlar aniqlandi.  
- **Model baholash (Model Evaluation)**: VotingClassifier modelining ROC AUC balli yuqori ekanligi aniqlangan.  
- **Kalibratsiya natijalari (Calibration Results)**: Kalibrlangan model aniqroq bashoratlar berdi (past Brier Score).  

---

### **Fayl tuzilmasi (File Structure)**  
- `dataset.csv`: Ma'lumotlar fayli (Dataset file)  
- `model_training.py`: Modelni o‘rgatish va baholash kodlari (Model training and evaluation script)  
- `README.md`: Loyihani tushuntirish fayli (This file)  

---

### **Natijaviy fayl (Final Output File)**  
Tugallangan model bashoratlari CSV faylga yozib chiqildi:  
```python
subm = pd.read_csv("sample_submission.csv")
subm['smoking'] = y_test_prob
subm.to_csv("final_submission.csv", index=False)
```  

---

Loyiha haqida savollaringiz bo‘lsa, [abdulboriyesonov339@gmail.com] orqali murojaat qilishingiz mumkin.  
