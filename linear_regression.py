import pandas as pd
import numpy as np
import os
import json 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
from scipy import stats

DATA_PATH = ''
TARGET_COL = "target"
TEST_SIZE = 0.2
SEED = 42
APPLY_BOXCOX = True
FEATURE_COLS_FOR_BOXCOX = ['gray', 'green', 'blue', 'peak', 'std', 'kurt', 'skew', 'ssim', 'mse']

OUTPUT_PATH = ''
OUTPUT_METRICS = os.path.join(OUTPUT_PATH, 'linear_metrics.json')
OUTPUT_PREDICTIONS = os.path.join(OUTPUT_PATH, 'predictions.xlsx')
RUN_BOOTSTRAP = True
BOOTSTRAP_ITERATIONS = 1000

def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def boxcox_on_train_data(X_train, cols):
    """Fitting Box-Cox transformation on selected columns in X_train dataset"""
    params = {}
    Xbc = X_train.copy()
    for col in cols:
        if col not in Xbc.columns:
            print(f"Warning:Box-Cox column '{col}' not found in training data.")
            continue
    col_vals = Xbc[col].astype(float).values
    shift = 0.0
    minv = np.nanmin(col_vals)
    if not np.isfinite(minv) or minv <= 0:
        shift = 1 - minv if np.isfinite(minv) else 1.0

    positive_vals = col_vals + shift

    try:
        bc_vals, lm = stats.boxcox(positive_vals)
        Xbc[col] = bc_vals
        params[col] = {"shift": float(shift), "lambda": float(lm)}
    except Exception as e:
        print(f"Could not apply Box-Cox to {col}: {e}")
    return Xbc, params

def apply_boxcox(X, params):
    """Apply pre-calculated Box-Cox transformation from above function"""
    X_out = X.copy()
    for col, p in params.items():
        if col not in X_out.columns:
            continue
        vals = X_out[col].astype(float).values + p["shift"]
        lm = p["lambda"]
        if lm != 0:
            X_out[col] = (np.power(vals,lm) - 1.0) / lm
        else:
            X_out[col] = np.log(vals)
    return X_out

def compute_aic_bic(y_true, y_pred, k):
    """Calculate AIC and BIC"""
    n = len(y_true)
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    if mse == 0: return -np.inf, -np.inf
    log_likelihood = -0.5 * n * np.log(2*np.pi*mse)-(1/(2*mse))*np.sum(residuals**2)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n)- 2 * log_likelihood
    return aic, bic 

def bootstrap_r2_ci(X_train, y_train, X_test, y_test, n_iter = 1000):
    """Performs bootstrap for linear regression model.
    Box-Cox transformation is re-fitted on each bootstrap sample."""

    r2_scores = []
    rmse_scores = []
    aic_scores = []
    bic_scores = []
    features_names = X_train.columns.tolist()

    for i in range(n_iter):
        X_resample, y_resample = resample(X_train, y_train)
        X_resample_bc, bc_params = boxcox_on_train_data(X_resample, y_resample)
        X_test_bc = apply_boxcox(X_test, bc_params)

        try: 
            pipeline_boot = Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', LinearRegression())
            ])
            pipeline_boot.fit(X_resample_bc, y_resample)

            y_pred_boot = pipeline_boot.predict(X_test_bc)
            score = r2_score(y_test, y_pred_boot)
            r2_scores.append(score)
            rmse_score = calculate_rmse(y_test,y_pred_boot)
            rmse_scores.append(rmse_score)
            k = 5
            aic, bic = compute_aic_bic(y_test, y_pred_boot, k)
            aic_scores.append(aic)
            bic_scores.append(bic)
        except Exception:
            continue

    ci = 0.05 # 95% confidence interval
    lower_r2 = np.percentile(r2_scores, 100 * (ci/2))
    upper_r2 = np.percentile(r2_scores, 100 * (1- ci/2))
    lower_rmse = np.percentile(rmse_scores, 100 * (ci/2))
    upper_rmse = np.percentile(rmse_scores, 100 * (1 - ci/2))
    lower_aic = np.percentile(aic_scores, 100 * (ci/2))
    upper_aic = np.percentile(aic_scores, 100 * (1 - ci/2))
    lower_bic = np.percentile(bic_scores, 100 * (ci/2))
    upper_bic = np.percentile(bic_scores, 100 * (1 - ci/2))

    return lower_r2, upper_r2, lower_rmse, upper_rmse, lower_aic, upper_aic, lower_bic, upper_bic

try:
    df = pd.read_excel(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    
    y = df[TARGET_COL].values.astype(float)
    X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).copy()
    feature_names = X.columns.tolist()
    if not feature_names:
        raise ValueError("No numeric feature columns found.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = SEED)

    if APPLY_BOXCOX:
        X_train_bc, bc_params = boxcox_on_train_data(X_train, feature_names)
        X_test_bc = apply_boxcox(X_test, bc_params)
    else:
        X_train_bc, X_test_bc = X_train.copy(), X_test.copy()
        bc_params = {}

    scaler = RobustScaler()
    X_train_std = scaler.fit_transform(X_train_bc)
    X_test_std = scaler.transform(X_test_bc)

    model = LinearRegression()
    model.fit(X_train_std, y_train)
    y_pred_test = model.predict(X_test_std)

    r2 = r2_score(y_test, y_pred_test)
    
    num_params = X_train_std.shape[1] + 1
    aic, bic = compute_aic_bic(y_test, y_pred_test, num_params)

    bootstrap_results = {}
    if RUN_BOOTSTRAP:
        print("\n --- Running BootStrap Analysis --- ")
        r2_lower, r2_upper, r2_values, rmse_lower, rmse_upper, rmse_values, aic_lower, aic_upper, aic_values, bic_lower, bic_upper, bic_values = bootstrap_r2_ci(
            X_train, y_train, X_test, y_test, n_iter=BOOTSTRAP_ITERATIONS
        )
        bootstrap_results = {
            "r2_mean": np.mean(r2_values), 
            "r2_95_ci": [r2_lower, r2_upper], 
            "rmse_mean": np.mean(rmse_values),
            "r2_95_ci": [rmse_lower, rmse_upper],
            "aic_mean": np.mean(aic_values),
            "aic_95_ci": [aic_lower, aic_upper],
            "bic_mean": np.mean(bic_values),
            "bic_95_ci": [bic_lower, bic_upper]
        }

        metrics_to_save = {
            "training_setup": {"features": feature_names, "used_boxcox": APPLY_BOXCOX, "scaler": "RobustScaler"},
            "evaluation_metrics": {"r2":r2, "aic":aic, "bic":bic},
            "bootstrap_results": bootstrap_results if RUN_BOOTSTRAP else "Not run",
            "model_parameters": {"coefficients": {name:float(coef) for name, coef in zip(feature_names, model.coef_)}, "intercept": float(model.intercept_)}
        }

        with open(OUTPUT_METRICS, "w") as f: json.dump(metrics_to_save, f, indent = 4)

        joblib.dump(model, os.path.join(OUTPUT_PATH, "linear_model.joblib"))
        joblib.dump(scaler, os.path.join(OUTPUT_PATH, "scaler.joblib"))
        with open(os.path.join(OUTPUT_PATH, "boxcox_params.json"), "w") as f: json.dump(bc_params, f, indent = 4)

        predictions_df = pd.DataFrame({'actual_value': y_test, 'predicted_value': y_pred_test})
        predictions_df.to_excel(OUTPUT_PREDICTIONS, index=False)

except FileNotFoundError:
    print(f"Error: The file was not found at '{DATA_PATH}'")