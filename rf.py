import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocessing import results

# Prepare features and target
X = results.drop('price', axis=1)
y = results['price']

# Don't brick the program
inf_mask = np.isinf(X).any(axis=1)
nan_mask = np.isnan(X).any(axis=1)

# Remove rows with infinite or NaN values
valid_mask = ~(inf_mask | nan_mask)
X_clean = X[valid_mask]
y_clean = y[valid_mask]

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.1, random_state=42)

# Model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\nTraining RMSE: ${train_rmse}")
print(f"Test RMSE: ${test_rmse}")
print(f"Training R²: {train_r2}")
print(f"Test R²: {test_r2}")
print(f"Test MAE: ${test_mae}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_clean.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Top 10)")
print(feature_importance.head(10).to_string(index=False))

# Check for overfitting
overfit_ratio = train_r2 / test_r2
print(f"\nTrain R^2 / Test R^2 ratio: {overfit_ratio}")
if overfit_ratio > 1.1:
    print("Model might be overfitting")
else:
    print("Model generalization looks fine")