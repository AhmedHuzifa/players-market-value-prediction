# players-market-value-prediction
Predict the market value (EUR) of Premier League football players based on match statistics and metadata. The final solution trains an XGBoost regression model and serves predictions through a FastAPI web service that can be run locally via "uv" or containerized with Docker for cloud deployment (e.g., AWS)

## Dataset

- File: `players-stats.csv`
- Rows: 413
- Target: `value_euros` (market value in EUR)

---

## Exploratory Data Analysis (EDA)

### Missing values
Only two columns contained missing values:

- `contract expiration` missing rate: 4.35%
- `other positions` missing rate: 60.29%

Decisions:
- Rows missing `contract expiration` were within a reasonable value range, so they were dropped.
- `contract expiration` was dropped afterwards because it is represented by `years_remaining`.
- `born` was dropped because it is represented by `age`.
- `other positions` was excluded from modeling due to the high missing rate.

### Target distribution
`value_euros` has a **long-tail distribution**, so the model is trained on a log scale:

- Train target: `log1p(value_euros)`
- Inference conversion: `expm1(prediction)`

### Redundant / overlapping columns
Multiple columns provide overlapping information:

- `g+a` is derived from `gls` and `ast`
- `gls` overlaps with `g-pk`, `pk`, `pkatt`
- `xg`, `npxg`, `xag`, `npxg+xag` are related expected performance metrics  
  → use `npxg+xag` as the compact expected contribution feature
- Per-90 metrics exist, but feeding them directly can be misleading; it can be better for the model to learn relations from totals + minutes.

### Correlation with target (numerical features)
Highest correlations with `value_euros`:

- `years_remaining`: **+0.51**
- `age`: **−0.46**
- `prgc`: **+0.45**
- `npxg+xag`: **+0.44**

### Categorical feature importance
Categorical importance analysis showed:

- `team`: **0.215234**
- `pos`: **0.072468**
- `nation`: **0.036018**

`nation` was therefore not included in the final model.

---

## Final Feature Set

The final model uses:

- **Numerical:** `age`, `years_remaining`, `prgc`, `npxg+xag`
- **Categorical:** `team`, `pos`

---

## Model Training & Evaluation

Models evaluated with **R²**, **MAE**, **RMSE**:

| Model             | R² (test) | MAE (test) | RMSE (test) | R² (train) | MAE (train) | RMSE (train) |    ΔR² |
| ----------------- | --------: | ---------: | ----------: | ---------: | ----------: | -----------: | -----: |
| Linear Regression |    0.4826 |     0.4435 |      0.5557 |     0.4947 |      0.3986 |       0.5129 | 0.0121 |
| Random Forest     |    0.3510 |     0.3861 |      0.5292 |     0.9241 |      0.1693 |       0.2138 | 0.5731 |
| Decision Tree     |    0.1705 |     0.6103 |      0.8587 |     1.0000 |      0.0000 |       0.0000 | 0.8295 |

Takeaway: Linear Regression generalizes better than RF/DT (which overfit heavily).

XGBoost performed best:

Tuned hyperparameters:

learning_rate = 0.01
max_depth = 3
n_estimators = 3000

| Stage                   |         R² |    MAE |   RMSE |
| ----------------------- | ---------: | -----: | -----: |
| XGBoost (before tuning) |     0.5410 | 0.3333 | 0.4666 |
| XGBoost (final test)    | **0.6577** | 0.3963 | 0.4975 |


The trained artifact is saved as:
- **`model.bin`** (contains the fitted encoder + trained model)

---

## Repository Structure

- `players-stats.csv` — dataset
- `train.ipynb`contains the overall EDA, training of different models and evaluating them, and tuning the hyperparametersof the best model.
- `train.py` — clean training script (final pipeline; no EDA / no model comparison)
- `predict.py` — loads `model.bin` and serves the FastAPI prediction API
- `test.py` — sends a sample request to the running API for quick verification
- `model.bin` — trained model artifact (encoder + model)
- `pyproject.toml` — dependencies (managed by `uv`)
- `Dockerfile` — container image definition

---

## Dependencies

Dependencies are defined in `pyproject.toml`.  
Using `uv`, installation and environment management are reproducible and simple.

---

## Quickstart

### 1) Run the API service locally with uv
**uv run python predict.py**
The service will run on:
http://localhost:8080/docs

### 2) Test the running service

In a separate terminal:
**uv run python test.py**

### 3) Docker
- Build the image
**docker build -t value-prediction .**

-Run the container
**docker run -it --rm -p 8080:8080 value-prediction**
The API will be available at:
http://localhost:8080

Again, it can be tested by running **uv run python test.py**


### 4) Deployment
The Docker image can be deployed to the cloud (e.g., AWS) by pushing it to a container registry and running it on a managed container service.
