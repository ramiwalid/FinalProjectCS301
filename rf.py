import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocessing import results

# Prepare features
X = results.drop('price', axis=1)
y = results['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduced complexity model to prevent overfitting
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42
)

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R^2 scores: {cv_scores}")
print(f"Mean CV R^2: {cv_scores.mean():.4f}")

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

print(f"\nTraining RMSE: ${train_rmse:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")
print(f"Training R^2: {train_r2:.4f}")
print(f"Test R^2: {test_r2:.4f}")
print(f"Test MAE: ${test_mae:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Top 10)")
print(feature_importance.head(10).to_string(index=False))

# Overfitting check
overfit_ratio = train_r2 / test_r2 if test_r2 > 0 else float('inf')
rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')

print(f"\nTrain R^2 / Test R^2 ratio: {overfit_ratio:.4f}")
print(f"Test RMSE / Train RMSE ratio: {rmse_ratio:.4f}")
print(f"CV Mean R^2: {cv_scores.mean():.4f}")

if overfit_ratio > 1.15:
    print("\nModel is probably overfitting")
else:
    print("\nModel generalization looks fine")