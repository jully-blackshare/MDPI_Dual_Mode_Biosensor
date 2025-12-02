# MDPI_Dual_Mode_Biosensor
Data files for analysis using colorimetric, photothermal and dual-mode sensing

**Portable Dual-Mode Biosensor for Quantitative Lateral Flow Assays Using Machine Learning and Smartphone Integration"**

##Python Scripts

### 1. 'linear_regression.py'
- Applies Box-Cox Transformation to input data
- Runs boostrap to calculate evaluation metrics and their confidence intervals
- Implements linear regression models
- Saves predictions, model information, and settings to output file 

### 2. 'linear_regression_lasso.py'
- Applies Box-Cox Transformation to input data
- Runs boostrap to calculate evaluation metrics and their confidence intervals
- Implements LASSO regularization to linear regressio model
- Saves predictions, model information, and settings to output file 


### 3. 'make_predictions.py'
- Based on trained model data, makes predictions with new input features
- Applies Box-Cox transformation based on pre-trained data

### 4. 'color_data.xlsx'
- Includes processed ratio for three color channels: gray, green, and blue for each Salmonella concentration used in regression analysis
- Columns:
  - gray, green, blue: intensity values extracted from line intensity analysis 
  - target: log-scale concentrations 

### 5. 'pt_data.xlsx'
- Includes features used to train photothermal dataset 
- Columns:
  - peak, std, kurt, skew, ssim, mse: features extracted for training 
  - target: log-scale concentrations

### 6. 'data.xlsx'
- Included merged features from color and photothermal dataset
- Columns:
  - gray, green, blue, peak, std, kurt, skew, ssim, mse
  - target: log-scale concentrations



