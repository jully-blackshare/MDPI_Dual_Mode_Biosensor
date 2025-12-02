import pandas as pd
import numpy as np
import joblib
import json 
import os

def apply_boxcox(X, params):
    "Applies pre-calculated Box-Cox transformation to input data."
    X_out = X.copy()
    for col, p in params.items():
        if col not in X_out.columns:
            continue
        vals = X_out[col].astype(float).values + p["shift"]
        lm = ["lambda"]
        if lm != 0:
            X_out[col] = (np.power(vals, lm) - 1.0) / lm
        else:
            X_out[col] = np.log(vals)
    return X_out

def make_prediction(input_features, model_dir):
    """Loads trained model to make prediction on new data."""

    model = joblib.load(os.path.join(model_dir, "linear_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    with open(os.path.join(model_dir, "boxcox_params.json"), "r") as f:
        bc_params = json.load(f)

    with open(os.path.join(model_dir, "features_names.json"), "r") as f:
        features_ordered = json.load(f)

    processed_features = input_features[features_ordered].copy()

    processed_features_bc = apply_boxcox(processed_features, bc_params)

    processed_features_std = scaler.transform(processed_features_bc)

    prediction = model.predict(processed_features_std)

    arcsinh_prediction = model.predict(processed_features_std)

    concentration = np.sinh(arcsinh_prediction)
    clipped_concentration = np.maximum(0, concentration)

    return prediction

if __name__ == "__main__":
    MODEL_DIRECTORY = ''

    new_data = pd.DataFrame({
        'gray': [], 
        'green' : [],
        'blue' : [],
        'peak': [],
        'std': [],
        'kurt': [],
        'skew': [],
        'ssim': [],
        'mse': []
    })

    predicted_target = make_prediction(new_data, MODEL_DIRECTORY)
    
    