import pandas as pd
import kagglehub
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,  mean_absolute_percentage_error
X_train_scaled, X_test_scaled, y_train, y_test = joblib.load("prepared_data.pkl")
import numpy as np

path = kagglehub.dataset_download("patelris/crop-yield-prediction-dataset")
df = pd.read_csv(f"{path}/yield_df.csv")

def evaluate(model, X_te, y_te, name):
    pred = model.predict(X_te)
    return{
        'Model': name,
        'MAE': mean_absolute_error(y_te, pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, pred)),
         'R2': r2_score(y_te, pred),
        "MAPE": mean_absolute_percentage_error(y_te, pred) * 100,
    }

# 1. Linear Regression
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)

joblib.dump(lin_model, "linear_regression_model.pkl")
print("Linear Regression model saved as linear_regression_model.pkl")

# 2. Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
import os

random_forest = RandomForestRegressor(n_estimators=200,
                                      max_depth=None, random_state=42,
                                      n_jobs= -1)
random_forest.fit(X_train_scaled, y_train)

joblib.dump(random_forest, "random_forest.pkl")
print("Random Forest saved as random_forest.pkl")

# 2.1 Hyperparameter tuning for Random Forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

rf_param_grid = {'n_estimators': [100,200,300],
              'max_depth': [None,20,40],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2]
              }

rf_grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    scoring= 'neg_root_mean_squared_error',
    cv = 3,
    verbose=2,
    n_jobs=-1
)
if os.path.exists('rf_grid_result.pkl'):
    rf_grid_search = joblib.load('rf_grid_result.pkl')
    print("Loaded saved tuning result.")
else:
    # first time — run the search, then save it
    rf_grid_search.fit(X_train_scaled, y_train)
    joblib.dump(rf_grid_search, 'rf_grid_result.pkl')
    print("Tuning done and saved.")

best_rf = rf_grid_search.best_estimator_
print("Best settings:", rf_grid_search.best_params_)

# 3. Gradient Boost Regression

from sklearn.ensemble import GradientBoostingRegressor

gbr_model= GradientBoostingRegressor(n_estimators=200, max_depth=5,
                    learning_rate=0.1, random_state=42)

gbr_model.fit(X_train_scaled, y_train)

joblib.dump(gbr_model, "gbr_model.pkl")
print("Gradient Boosting Regressor model saved as gbr_model.pkl")

# 3.1 Hyperparameter tuning for Gradient Boosting Regressor

gbr_param_grid = {'n_estimators': [100,200,300],
              'max_depth': [3,5,7],
              'learning_rate': [0.05,0.1,0.2],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2]
              }

gbr_grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=gbr_param_grid,
    scoring= 'neg_root_mean_squared_error',
    cv = 3,
    verbose=2,
    n_jobs=-1
)
# saving tuning results
if os.path.exists('gbr_grid_result.pkl'):
    gbr_grid_search = joblib.load('gbr_grid_result.pkl')
    print("Loaded saved tuning result.")
else:
    gbr_grid_search.fit(X_train_scaled, y_train)
    joblib.dump(gbr_grid_search, 'gbr_grid_result.pkl')
    print("Tuning done and saved.")

best_gbr = gbr_grid_search.best_estimator_
print("Best settings:", gbr_grid_search.best_params_)

# 3. Model Comparsion
results = pd.DataFrame([
    evaluate(lin_model, X_test_scaled,y_test, "Linear(baseline)"),
    evaluate(random_forest,  X_test_scaled, y_test, "Random Forest (untuned)"),
    evaluate(best_rf,        X_test_scaled, y_test, "Random Forest (tuned)"),
    evaluate(gbr_model,      X_test_scaled, y_test, "Gradient Boosting (untuned)"),
    evaluate(best_gbr,       X_test_scaled, y_test, "Gradient Boosting (tuned)"),
])

results = results.sort_values("R2", ascending=False).round(3)

print("Model Comparison")
print(results.to_string(index=False))

results.to_csv("model_comparison.csv", index=False)
print("\nSaved to model_comparison.csv")

# 4. Results Visualized
results = X_test_scaled.copy()

# 4.1 Each country best crop based on predicted yield
results['Area'] = df.loc[X_test_scaled.index, 'Area']
results['Item'] = df.loc[X_test_scaled.index, 'Item']
results['Predicted_Yield'] = random_forest.predict(X_test_scaled)
results['Actual_Yield'] = y_test

best_crop = results.loc[
    results.groupby('Area')['Predicted_Yield'].idxmax()
]
print(best_crop[['Area', 'Item', 'Predicted_Yield', 'Actual_Yield']]
      .sort_values('Predicted_Yield', ascending=False)
      .head(20))
results['Error'] = results['Actual_Yield'] - results['Predicted_Yield']

# 4.2 RMSE per crop, how well the model does on each crop
rmse_per_crop = (
    results.groupby('Item')
    .apply(lambda g: np.sqrt(mean_squared_error(g['Actual_Yield'], g['Predicted_Yield'])))
    .reset_index(name='RMSE')
    .sort_values('RMSE', ascending=False)
)

print(rmse_per_crop)
plt.figure(figsize=(10, 5))
plt.bar(rmse_per_crop['Item'], rmse_per_crop['RMSE'])
plt.xticks(rotation=45, ha='right')
plt.title('RMSE per Crop (Random Forest)')
plt.xlabel('Crop')
plt.ylabel('RMSE (hg/ha)')
plt.tight_layout()
plt.show()

# 4.3 Scatter plot for actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(results['Actual_Yield'], results['Predicted_Yield'], alpha=0.3, s=10)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Predicted vs Actual (Random Forest)')
plt.legend()
plt.tight_layout()
plt.show()