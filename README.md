# Insurance Charges Prediction using Linear Regression

## 1) Task objective

The objective of this project is to build a **regression model** that predicts a person’s **medical insurance charges** from demographic and lifestyle attributes in the dataset.

The notebook focuses on:

- Understanding how input features (age, sex, BMI, number of children, smoking status, and region) relate to insurance charges.
- Training a **Linear Regression** model to estimate a **continuous target variable** (`charges`).
- Evaluating model quality using standard regression metrics:
  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Squared Error)**
  - **R² (Coefficient of Determination)**
- Visualizing major feature-target relationships to support interpretation.

---

## 2) My approach

I followed a structured end-to-end machine learning workflow:

### a) Data loading and initial inspection
- Loaded `insurance.csv` into a pandas DataFrame.
- Checked:
  - Dataset shape: **1338 rows × 7 columns**
  - Data types of all columns
  - Descriptive statistics for numerical columns
  - Missing values (`isnull().sum()`) → **no missing values found**

### b) Preprocessing
- Created a copy of the original data for modeling.
- Encoded categorical variables using `LabelEncoder`:
  - `sex` (female/male)
  - `smoker` (no/yes)
  - `region` (4 regional categories)
- Printed mapping for interpretability, e.g.:
  - `smoker`: `no -> 0`, `yes -> 1`
  - `sex`: `female -> 0`, `male -> 1`

### c) Feature-target split
- Defined:
  - **Features (X):** all columns except `charges`
  - **Target (Y):** `charges`

### d) Train-test split
- Split data into:
  - **80% training set** (`1070` records)
  - **20% test set** (`268` records)
- Used `random_state=42` for reproducibility.

### e) Model training
- Trained a `LinearRegression` model from scikit-learn on the training data.

### f) Prediction and evaluation
- Generated predictions for both training and test sets.
- Computed:
  - MAE
  - RMSE
  - R²
- Compared train and test metrics to assess generalization.

### g) Visualization
Created exploratory scatter plots for:
1. **BMI vs Charges**
2. **Age vs Charges**
3. **Smoking Status vs Charges**

These plots help explain linear/nonlinear trends and feature impact on medical charges.

---

## 3) Results and insights

### Model performance

**Training set**
- **MAE:** \$4,208.76  
- **RMSE:** \$6,105.79  
- **R²:** 0.7417  

**Test set**
- **MAE:** \$4,186.51  
- **RMSE:** \$5,799.59  
- **R²:** 0.7833  

### Key insights

- The model explains around **74–78% of variance** in insurance charges, which is strong for a baseline linear model.
- **Train and test metrics are close**, indicating good generalization and no obvious overfitting.
- From visual exploration:
  - **Smoking status** shows a strong effect on charges (smokers tend to have much higher costs).
  - **Age** generally has a positive relationship with charges.
  - **BMI** also contributes to charge variation, though with dispersion.
- A few predictions are negative in output, which can happen with unconstrained linear regression and suggests room for model refinement.

### Potential improvements (next steps)

- Use **One-Hot Encoding** for categorical variables (especially `region`) instead of label encoding to avoid unintended ordinal assumptions.
- Try advanced models (e.g., **Random Forest Regressor**, **XGBoost**, **Gradient Boosting**) for potentially better accuracy.
- Apply feature engineering (e.g., interaction terms like `smoker × age`, `bmi × smoker`).
- Consider target transformation (e.g., `log(charges)`) to reduce skew and improve error behavior.

---
**Conclusion:**  
This project successfully demonstrates a complete regression pipeline and delivers a reliable baseline model for predicting insurance charges using interpretable features and clear evaluation.
