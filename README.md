# Used Vehicle Price Prediction (Rusty Bargain App)

An end-to-end Machine Learning production pipeline that evaluates, benchmarks, and optimizes high-performance regression architectures to accurately predict used car market values. Built as a scalable core service for the *Rusty Bargain* mobile application.

---

## 📌 Project Overview & Goals
Rusty Bargain is a used car marketplace developing an app to attract new customers by offering rapid, reliable valuation metrics. Users receive immediate market value estimations based on physical vehicle specifications, historic depreciation trends, and structural trim options.

Commercial viability dictates a balanced optimizations strategy across three business pillars:
1. **Prediction Quality:** Minimizing the Root Mean Squared Error (RMSE) to capture customer trust.
2. **Prediction Speed:** Ensuring near-instantaneous, low-latency execution loops inside mobile endpoints.
3. **Training Time:** Maintaining an efficient, light computational profile for continuous pipeline retraining with new real-time market data.

---

## 📂 Project Structure
```text
├── data/
│   └── car_data.csv                       # Historical used vehicle listings database
├── notebooks/
│   └── used_vehicle_price_prediction.ipynb # EDA, data preparation pipelines, and model evaluation
├── src/
│   └── funciones_personales.py             # Reusable helpers for missing-value analytics & profiling
├── environment.yml                         # Production Conda environment manifest
└── requirements.txt                        # Lightweight Pip environmental dependency definition
```

---

## 🛠️ Data Preprocessing & Strategy
To protect structural optimization phases from **Data Leakage**, all validation parameters were fitted solely on the training matrices before scaling data down the stream.

* **Data Cleaning & Filtering:** Replaced structural missing categorical records with `'unknown'`. Filtered out anomalous registration parameters outside the historical 1910–2019 baseline. Replaced standard error indices (like engine power values at `0`) with validation medians. Dropped rows featuring unrealistic pricing markers under \$500 (approx. 10.17% of total records) to safeguard target distributions against non-functional synthetic rows.
* **Feature Engineering:** Built a synthetic `VehicleAge` feature (`2019 - RegistrationYear`) to model physical economic depreciation maps explicitly, dropping the collinear registration attribute afterwards.
* **Encoding Optimization:** Categorical values were segmented across dynamic processing tracks tailored to the target algorithms:
  * **Ensemble Tree Parsers (LightGBM / CatBoost):** Handled via explicit internal category typing to leverage native split optimization algorithms.
  * **Matrix Boosters (Random Forest / XGBoost):** Expanded using dense One-Hot Encoding framework schemas.
  * **Parametric Estimators (Linear Regression Baseline):** Standardized using strict One-Hot maps with `drop='first'` coupled with a robust `StandardScaler` wrapper to eliminate collinear dependency matrices.

---

## 🚀 Model Comparison & Benchmarks
All algorithms were trained on a fixed 70% data allocation split, optimized on a isolated 15% validation matrix, and locked against a remaining 15% testing tier (`random_state=54321`):

| Model | Hyperparameters / Configuration | Validation RMSE | Training Time (s) | Prediction Time (s) |
| :--- | :--- | :---: | :---: | :---: |
| **LightGBM** | `n_estimators=400, learning_rate=0.05` | **1642.75** | **10.21** | **0.70** |
| LightGBM | `n_estimators=200, learning_rate=0.10` | 1647.50 | 6.53 | 0.38 |
| XGBoost | `n_estimators=400, learning_rate=0.05, max_depth=8` | 1655.91 | 946.35 | 0.53 |
| XGBoost | `n_estimators=200, learning_rate=0.10, max_depth=6` | 1715.73 | 354.31 | 0.32 |
| Random Forest | `n_estimators=100, max_depth=15` | 1746.20 | 127.77 | 0.38 |
| Random Forest | `n_estimators=100, max_depth=10` | 1969.91 | 100.71 | 0.20 |
| Linear Regression | `Default (Baseline Model)` | 2852.66 | 7.85 | 0.01 |

> *Note: CatBoost was profiled inside independent compute instances, yielding an approximate benchmark RMSE baseline of ~1700.*

---

## 🏆 Final Model Evaluation (Winner)
**LightGBM** (`n_estimators=400, learning_rate=0.05, random_state=54321`) was deployed as the winning production candidate. It consistently outmatched alternative gradient tree architectures across execution speed, scaling footprint, and optimization performance.

Final verification on the entirely unseen **Test Set**:
* **Test RMSE:** `1649.62` (A negligible ~0.4% deviation from validation metrics, verifying zero overfit)
* **Test Prediction Latency:** `0.67 seconds`

### 💡 Core Takeaways
* **Server Scalability:** While XGBoost achieved a comparable predictive boundary, it required **946.35 seconds (~15.7 minutes)** to converge. LightGBM reached an even lower error state in a mere **10.21 seconds**.
* **Commercial Value:** The microsecond inference pipeline footprint guarantees stable response times for concurrent app operations while maintaining minimal infrastructure overhead costs.

---

## 💻 Local Setup & Execution

Python 3.9 is the targeted environment configuration for cross-platform local reproducibility.

### Option 1: Conda Environment Setup (Recommended)
```bash
# Clone repository and enter workspace
cd rusty-bargain-predictor

# Create environment from the environmental manifest
conda env create -f environment.yml

# Activate environment
conda activate used-vehicles-price-prediction
```

### Option 2: Standalone Pip Virtual Environment
```bash
# Create and activate environment instance
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate      # On Windows PowerShell

# Standardize dependency framework
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Running the Evaluation Notebook
Launch the runtime instance targeting the internal notebooks workspace root to ensure data parsing relative paths resolve properly:
```bash
jupyter lab --notebook-dir=notebooks
```
Open `used_vehicle_price_prediction.ipynb` and run the cells sequentially. 

*Hardware Note: Running the dense `Random Forest` and `XGBoost` cross-validation loops can be highly resource-intensive on low-tier personal workstations; parameters inside those specific search blocks can be safely scaled down to accelerate verification if required.*
