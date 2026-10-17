# Used Vehicle Price Prediction

> Versión en español: [LEEME.md](LEEME.md)

A regression project that predicts **the market value of used cars** from their technical specifications, for the valuation feature of a used-car app.

The project cleans 354,369 car listings, builds a vehicle-age feature, and compares **Linear Regression, Random Forest, XGBoost, and LightGBM** on three criteria: prediction quality (RMSE), training time, and prediction time. The selected model, **LightGBM (400 trees, learning rate 0.05)**, achieved an **RMSE of 1,642.76 on validation** and **1,649.63 on the independent test set**, the lowest error of all models while training in seconds instead of minutes.

---

# Business Context

Rusty Bargain, a used-car marketplace, is building an app where users can quickly find out the market value of their car. The company cares about three things:

| Criterion | Why it matters |
| --- | --- |
| **Prediction quality** | Fair, realistic valuations build user trust |
| **Prediction speed** | The app must answer immediately |
| **Training time** | The model should be easy to retrain with new listings |

The model must predict one target:

| Target | Description |
| --- | --- |
| `Price` | Listing price in euros |

---

# Key Concepts

| Term | Meaning |
| --- | --- |
| Vehicle age | `2019 − RegistrationYear`; it represents depreciation more directly than the year itself |
| One-Hot Encoding | Turns each category into a 0/1 column; needed by Random Forest, XGBoost, and Linear Regression |
| Native categorical handling | LightGBM can split directly on `category` columns, without One-Hot Encoding |
| Gradient boosting | Builds trees one after another, each correcting the errors of the previous ones (LightGBM, XGBoost) |
| Baseline | A simple model (here, Linear Regression) used to check that the complex models add value |

---

# Evaluation Metric

The main metric is the **Root Mean Squared Error (RMSE)**:

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

RMSE is in the same unit as the price (euros) and penalizes large errors more than small ones, which matters because a very wrong valuation hurts user trust the most. **Training time** and **prediction time** were measured with `time.perf_counter()` for every model.

---

# Dataset

| File | Rows | Columns | Description |
| --- | ---: | ---: | --- |
| `data/car_data.csv` | 354,369 | 16 | Used-car listings collected in March–April 2016 |

| Column | Description |
| --- | --- |
| `Price` | Price in euros (target) |
| `VehicleType` | Body type |
| `RegistrationYear`, `RegistrationMonth` | Registration date |
| `Gearbox` | Manual or automatic |
| `Power` | Power (hp) |
| `Model`, `Brand` | Model and brand |
| `Mileage` | Mileage (km) |
| `FuelType` | Fuel type |
| `NotRepaired` | Whether the car has unrepaired damage |
| `DateCrawled`, `DateCreated`, `LastSeen` | Dates related to the listing |
| `NumberOfPictures`, `PostalCode` | Number of pictures and owner's postal code |

**Data source.** The dataset was provided by **TripleTen** as part of its Data Science Bootcamp and is included in this repository for educational and reproducibility purposes.

---

# Project Workflow

```text
Raw data (354,369 listings)
            │
            ▼
1. Data preparation: missing values, out-of-range values, VehicleAge
            │
            ▼
2. Split: 70% train / 15% validation / 15% test
            │
            ▼
3. Model-specific preprocessing (imputation fitted on train)
   ├── LightGBM: native categorical columns
   ├── Random Forest and XGBoost: One-Hot Encoding
   └── Linear Regression: One-Hot Encoding (drop_first) + StandardScaler
            │
            ▼
4. Train and time each model with a small hyperparameter grid
            │
            ▼
5. Evaluate the best model once on the test set
```

---

# 1. Data Preparation

## Missing Values

| Column | Missing | Treatment |
| --- | ---: | --- |
| `NotRepaired` | 71,154 | `"unknown"` |
| `VehicleType` | 37,490 | `"unknown"` |
| `FuelType` | 32,895 | `"unknown"` |
| `Gearbox` | 19,833 | `"unknown"` |
| `Model` | 19,705 | `"unknown"` |

108,555 rows (30.6%) had at least one missing value, so filling them with `"unknown"` kept the data instead of dropping almost a third of it.

## Out-of-Range Values

| Column | Problem | Treatment |
| --- | --- | --- |
| `Price` | 36,054 listings (10.2%) under €500, including 10,772 with a price of 0 | Rows removed, so the model does not learn from unrealistic targets |
| `RegistrationYear` | Values from 1000 to 9999 | Years outside 1910–2019 set to missing and later imputed with the training median |
| `Power` | 40,225 listings with 0 hp | Replaced with the median power |
| `RegistrationMonth` | Month 0 | Treated as "unknown month" (0) |

## Feature Engineering and Selection

* **`VehicleAge` = 2019 − `RegistrationYear`**, and `RegistrationYear` was dropped.
* Removed columns: `DateCrawled`, `DateCreated`, `LastSeen` (listing dates), `NumberOfPictures` (always 0), and `PostalCode`.
* Final data: **318,315 rows** and **10 features** (6 categorical and 4 numeric).

---

# 2. Data Split and Preprocessing

```text
318,315 listings
│
├── 70% TRAIN ─────────→ 222,947 → model training
├── 15% VALIDATION ────→  47,748 → model and hyperparameter selection
└── 15% TEST ──────────→  47,620 → final evaluation (used only once)
```

The imputers, the One-Hot Encoder, and the scaler were **fitted on the training set only** and then applied to validation and test.

| Model family | Input |
| --- | --- |
| LightGBM | Numeric columns imputed + categorical columns as `category` (10 features) |
| Random Forest, XGBoost | Numeric columns imputed + One-Hot Encoding (317 features) |
| Linear Regression | One-Hot Encoding with `drop_first` + `StandardScaler` (311 features) |

---

# 3. Modeling

| Model | Configuration | Validation RMSE | Training time | Prediction time |
| --- | --- | ---: | ---: | ---: |
| **LightGBM** | `n_estimators=400, learning_rate=0.05` | **1,642.76** | 7 s | 0.72 s |
| LightGBM | `n_estimators=200, learning_rate=0.1` | 1,647.50 | 3 s | 0.31 s |
| XGBoost | `n_estimators=400, learning_rate=0.05, max_depth=8` | 1,655.91 | 911 s | 0.57 s |
| XGBoost | `n_estimators=200, learning_rate=0.1, max_depth=6` | 1,715.73 | 346 s | 0.34 s |
| Random Forest | `n_estimators=100, max_depth=15` | 1,746.21 | 123 s | 0.34 s |
| Random Forest | `n_estimators=100, max_depth=10` | 1,969.91 | 101 s | 0.18 s |
| Random Forest | `n_estimators=50, max_depth=10` | 1,972.17 | 53 s | 0.17 s |
| Linear Regression (baseline) | Default | 2,852.67 | 8 s | 0.01 s |

*Times from the notebook run; they change slightly between runs and machines. Prediction times are for the whole validation set (47,748 cars).*

* **Every tree-based model clearly beat the baseline** (RMSE 2,853), so the non-linear relationships between features and price matter.
* **LightGBM had the lowest error and was also among the fastest to train**: about 7 seconds, while XGBoost needed about 15 minutes to reach a slightly higher error.
* Random Forest improved with more depth, but it stayed behind the boosting models.
* **CatBoost** is included in the notebook but commented out, because it required more computing power than was available. In a separate run it reached an RMSE of about 1,700, which was not saved in the notebook.

## Final Result

The selected model was evaluated **once** on the test set:

```text
Model:            LightGBM (n_estimators=400, learning_rate=0.05)
Validation RMSE:  1,642.76
Test RMSE:        1,649.63
Prediction time:  0.68 s for 47,620 cars
```

The test error is only 0.4% higher than the validation error, so the model generalizes well to new listings.

---

# Conclusions

1. **Data quality required several decisions**: almost a third of the rows had missing values, 10% had unrealistic prices, and some registration years and power values were impossible.

2. **Gradient boosting gave the best results.** LightGBM and XGBoost had the lowest errors, and Linear Regression was far behind.

3. **LightGBM offered the best balance of the three business criteria**: the lowest RMSE, training in seconds, and fast predictions.

4. **The result is stable**: the test RMSE (1,649.63) is almost the same as the validation RMSE (1,642.76).

## Business Interpretation

LightGBM can value a car almost instantly and can be retrained quickly as new listings arrive. An average error of about €1,650 should be presented to users as an approximate value, not as an exact price.

## Limitations and Next Steps

* **Registration years after 2016:** the listings were collected in 2016, but 14,425 cars (4%) have a registration year between 2017 and 2019, which is not possible. Limiting the range to 2016 and computing the age relative to the listing date would make `VehicleAge` more accurate.
* **Power:** the 0-hp values were replaced with the median of the whole dataset before the split (a small data leak), and 302 values above 1,000 hp (up to 20,000) were not treated. The median should be computed on the training set only, and extreme values should be capped.
* Tune LightGBM further (number of leaves, minimum samples per leaf) with cross-validation.
* Run CatBoost with enough computing power and save its results.
* Analyze feature importance to explain which characteristics drive the price.

---

# Project Structure

```text
used-vehicles-price-prediction/
│
├── data/
│   └── car_data.csv
│
├── notebooks/
│   └── used_vehicle_price_prediction.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
└── README.md
```

`src/funciones_personales.py` contains helper functions to inspect missing values (`mostrar_nan`), train and time one model (`evaluar_modelo`), and test a list of hyperparameter combinations (`probar_hiperparametros`).

---

# Installation

INSTALLATION_PLACEHOLDER

---

# Technologies

* Python
* Pandas
* NumPy
* scikit-learn (Linear Regression, Random Forest, One-Hot Encoding, StandardScaler, SimpleImputer)
* LightGBM
* XGBoost
* CatBoost
* Jupyter Notebook

---

# Skills Demonstrated

* Regression with tabular data
* Data cleaning: missing values, out-of-range values, and unrealistic targets
* Feature engineering (vehicle age)
* Model-specific preprocessing (encoders and imputers fitted on the training set)
* Comparison of Linear Regression, Random Forest, XGBoost, and LightGBM
* Hyperparameter tuning
* Measuring training and prediction time
* Choosing a model by balancing accuracy and speed

---

## Author

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
