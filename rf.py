import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocessing import results
from sklearn.ensemble import GradientBoostingRegressor


# Prepare features and target
X = results.drop('price', axis=1)
y = results['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42
)

# Cross-validation randmo forest
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

print(f"\nRF Training RMSE: ${train_rmse:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")
print(f"Training R^2: {train_r2:.4f}")
print(f"Test R^2: {test_r2:.4f}")
print(f"Test MAE: ${test_mae:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRF Feature Importance (Top 10)")
print(feature_importance.head(10).to_string(index=False))

# Overfitting check
overfit_ratio = train_r2 / test_r2 if test_r2 > 0 else float('inf')
rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')

print(f"\nRF Train R^2 / Test R^2 ratio: {overfit_ratio:.4f}")
print(f"RF Test RMSE / Train RMSE ratio: {rmse_ratio:.4f}")
print(f"RF CV Mean R^2: {cv_scores.mean():.4f}")

if overfit_ratio > 1.15:
    print("\nRF Model is probably overfitting")
else:
    print("\nRF Model generalization looks fine")

# Gradient Boosting Model
gb_model = GradientBoostingRegressor(
    n_estimators=1000, 
    max_depth=3, 
    learning_rate=0.01, 
    random_state=42
)

# Cross-validation for GB
gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='r2')
print(f"\nGradient Boosting Cross-validation R^2 scores: {gb_cv_scores}")
print(f"GB Mean CV R^2: {gb_cv_scores.mean():.4f}")

gb_model.fit(X_train, y_train)

# Make predictions for GB
gb_pred_train = gb_model.predict(X_train)
gb_pred_test = gb_model.predict(X_test)

# Calculate metrics for GB
gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_pred_train))
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_pred_test))
gb_train_r2 = r2_score(y_train, gb_pred_train)
gb_test_r2 = r2_score(y_test, gb_pred_test)
gb_test_mae = mean_absolute_error(y_test, gb_pred_test)

print(f"\nGradient Boosting Results:")
print(f"Training RMSE: ${gb_train_rmse:.2f}")
print(f"Test RMSE: ${gb_test_rmse:.2f}")
print(f"Training R^2: {gb_train_r2:.4f}")
print(f"Test R^2: {gb_test_r2:.4f}")
print(f"Test MAE: ${gb_test_mae:.2f}")

# Feature importance for GB
gb_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nGB Feature Importance (Top 10)")
print(gb_feature_importance.head(10).to_string(index=False))

# Overfitting check for GB
gb_overfit_ratio = gb_train_r2 / gb_test_r2 if gb_test_r2 > 0 else float('inf')
gb_rmse_ratio = gb_test_rmse / gb_train_rmse if gb_train_rmse > 0 else float('inf')

print(f"\nGB Train R^2 / Test R^2 ratio: {gb_overfit_ratio:.4f}")
print(f"GB Test RMSE / Train RMSE ratio: {gb_rmse_ratio:.4f}")
print(f"GB CV Mean R^2: {gb_cv_scores.mean():.4f}")

if gb_overfit_ratio > 1.15:
    print("\nGB Model is probably overfitting")
else:
    print("\nGB Model generalization looks fine")