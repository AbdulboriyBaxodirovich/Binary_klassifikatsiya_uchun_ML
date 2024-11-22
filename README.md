## Dataset README

## Dataset haqida
Bu dataset **dehqonchilik va hosil samaradorligi**ni tahlil qilish uchun yaratilgan. Ushbu ma'lumotlar turli xil omillar, jumladan, iqlim sharoitlari va changlatish omillari asosida hosil (yield) miqdorini bashorat qilish uchun ishlatilishi mumkin.

## Eng yaxshi MAE = 244

## Ma'lumot ustunlari
Dataset quyidagi ustunlardan iborat:

### 1. **id**
   - **Turi**: Categorical
   - **Tavsifi**: Har bir qator uchun unikal identifikator.

### 2. **Row#**
   - **Turi**: Integer
   - **Tavsifi**: Qatorning tartib raqami.

### 3. **clonesize**
   - **Turi**: Numeric
   - **Tavsifi**: O'simlikning klonlash darajasini o'lchaydigan qiymat.

### 4. **honeybee**
   - **Turi**: Numeric
   - **Tavsifi**: Asalarilar soni (honeybee) va ularning changlatishda qatnashish darajasi.

### 5. **bumbles**
   - **Turi**: Numeric
   - **Tavsifi**: Yovvoyi asalarilar (bumbles) soni va ularning changlatishda ishtiroki.

### 6. **andrena**
   - **Turi**: Numeric
   - **Tavsifi**: Andrena turidagi changlatuvchi asalarilar soni.

### 7. **osmia**
   - **Turi**: Numeric
   - **Tavsifi**: Osmia turidagi changlatuvchi asalarilar soni.

### 8. **MaxOfUpperTRange**
   - **Turi**: Numeric
   - **Tavsifi**: Haroratning yuqori chegarasi.

### 9. **MinOfUpperTRange**
   - **Turi**: Numeric
   - **Tavsifi**: Yuqori harorat oralig'ining minimal qiymati.

### 10. **AverageOfUpperTRange**
   - **Turi**: Numeric
   - **Tavsifi**: Yuqori harorat oralig'ining o'rtacha qiymati.

### 11. **MaxOfLowerTRange**
   - **Turi**: Numeric
   - **Tavsifi**: Past harorat oralig'ining maksimal qiymati.

### 12. **MinOfLowerTRange**
   - **Turi**: Numeric
   - **Tavsifi**: Past harorat oralig'ining minimal qiymati.

### 13. **AverageOfLowerTRange**
   - **Turi**: Numeric
   - **Tavsifi**: Past harorat oralig'ining o'rtacha qiymati.

### 14. **RainingDays**
   - **Turi**: Numeric
   - **Tavsifi**: Yomg'irli kunlarning soni.

### 15. **AverageRainingDays**
   - **Turi**: Numeric
   - **Tavsifi**: Yomg'irli kunlarning o'rtacha soni.

### 16. **fruitset**
   - **Turi**: Numeric
   - **Tavsifi**: Hosil miqdori shakllanish darajasi.

### 17. **fruitmass**
   - **Turi**: Numeric
   - **Tavsifi**: Meva massasining o'rtacha qiymati.

### 18. **seeds**
   - **Turi**: Numeric
   - **Tavsifi**: O'simlik urug'lari soni.

### 19. **yield**
   - **Turi**: Numeric (Target)
   - **Tavsifi**: Olingan hosil miqdori (target ustun).

## Feature Engineering
Datasetda quyidagi yangi ustunlar feature engineering orqali qo'shildi:

### 1. Yuqori va past haroratlar farqlari:
   - `UpperTempRange`: Yuqori haroratlarning maksimal va minimal qiymatlari orasidagi farqni hisoblaydi.
   - `LowerTempRange`: Past haroratlarning maksimal va minimal qiymatlari orasidagi farqni hisoblaydi.

### 2. Haroratning o'rtacha farqi:
   - `AvgTempDifference`: O'rtacha yuqori harorat va o'rtacha past harorat orasidagi farqni aks ettiradi.

### 3. Changlatuvchilar faoliyati bo'yicha indeks:
   - `PollinatorActivity`: Asalari, bumblebee, andrena va osmia kabi changlatuvchilar faoliyatini jamlab indeks hosil qiladi.

### 4. Changlatuvchi turlar nisbati:
   - `HoneybeeToBumblesRatio`: Honeybee va bumbles o'rtasidagi nisbat.
   - `HoneybeeToAndrenaRatio`: Honeybee va andrena o'rtasidagi nisbat.
   - `HoneybeeToOsmiaRatio`: Honeybee va osmia o'rtasidagi nisbat.

### 5. Yomg'ir kunlari soni:
   - `TotalRainyDays`: Yomg'irli kunlarning umumiy sonini hisoblaydi (`RainingDays` va `AverageRainingDays` mahsuloti).

### 6. Yomg'ir intensivligi:
   - `RainIntensity`: Yomg'irning intensivligini ko'rsatadi (`AverageRainingDays` ni `RainingDays` ga bo'lish orqali hisoblanadi).

### 7. Urug' zichligi:
   - `SeedDensity`: Har bir meva massasiga to'g'ri keladigan urug'lar soni (`seeds`ni `fruitmass`ga bo'lish orqali aniqlanadi).

## Modellar
Bu dataset quyidagi regressiya modellarini sinab ko'rish uchun ishlatilgan:

1. **Chiziqli Modellar:**
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - ElasticNet Regression

2. **Ansambl Modellar:**
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost
   - LightGBM

### Ansambl Modellarni Optimizatsiya
Ansambl modellarni yanada samarali ishlashi uchun **Optuna** yordamida hiperparametrlarni tuning qilindi. Ushbu optimizatsiya algoritmi modellarning giperparametrlarini aniqlashda samaradorlikni oshirishga yordam beradi va yuqori aniqlikdagi bashoratlarni ta'minlaydi.

## Foydalanish
Yuqoridagi ustunlar va yangi yaratilgan xususiyatlar hosildorlikni yanada aniqroq bashorat qilishga yordam beradi. Ushbu dataset turli xil regression algoritmlar, shuningdek, ansambl modellari bilan tahlil qilish uchun ishlatilishi mumkin.
"""
