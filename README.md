# Used Vehicle Price Prediction

Machine learning project for Rusty Bargain, a used-car marketplace. The goal is
to predict a vehicle's market price from its technical specifications and
equipment while comparing prediction quality, training time, and prediction
speed.

## Project Contents

- `data/car_data.csv`: used-vehicle listings used by the notebook.
- `notebooks/used_vehicle_price_prediction.ipynb`: exploratory analysis,
	preprocessing, model training, comparison, and final evaluation.
- `src/funciones_personales.py`: reusable helpers for missing-value inspection
	and model evaluation.
- `requirements.txt`: pip-based environment definition.
- `environment.yml`: Conda environment definition.

## Models

The notebook compares:

- Linear Regression as a baseline.
- Random Forest Regressor.
- LightGBM Regressor.
- CatBoost Regressor.
- XGBoost Regressor.

The selected model is LightGBM. In the saved benchmark, the best validation
configuration used `n_estimators=400`, `learning_rate=0.05`, and
`random_state=54321`, reaching a validation RMSE of approximately `1642.76`.
The final test evaluation reached an RMSE of approximately `1649.63`.

## Data Preparation

The notebook applies the following transformations:

- Replaces missing categorical values with `unknown`.
- Treats registration years outside 1910-2019 as missing.
- Replaces zero engine power with the training-set median.
- Treats registration month `0` as missing and imputes it separately.
- Removes listings with a price below 500.
- Drops crawl and location fields that are not used as model features.
- Creates `VehicleAge = 2019 - RegistrationYear` and removes the original year.
- Splits the data into approximately 70% training, 15% validation, and 15% test
	sets using `random_state=54321`.

Preprocessing parameters are fitted on the training data and then applied to
validation and test data to avoid data leakage. One-hot encoding is used for
Random Forest, XGBoost, and Linear Regression; LightGBM and CatBoost use the
categorical features directly.

## Setup

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate used-vehicles-price-prediction
```

### Option 2: Python virtual environment and pip

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\\Scripts\\activate      # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Notebook

From the repository root, start Jupyter with the notebook directory as its
working directory. This is required because the notebook uses relative paths
to `data/` and `src/`:

```bash
jupyter lab --notebook-dir=notebooks
```

Open `used_vehicle_price_prediction.ipynb` and run the cells in order. The
notebook contains separate model experiments. Random Forest and XGBoost can
take considerably longer and may require reducing their estimator counts or
depth on a lower-powered computer. CatBoost is included for comparison but may
also be computationally intensive.

## Reproducibility Notes

The benchmark times in the notebook were measured on the original machine and
will vary with hardware, operating system, thread settings, and library
versions. The data split and model random state are fixed where applicable, but
training and prediction times should not be treated as portable measurements.