# Crop-Yield-Predictor-Pipeline
A machine learning pipeline to predict crop yields (hg/ha) using climate and pesticide data.

Dataset: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset/data

Goal: Predict hg/ha_yield based on year, rainfall, pesticides, temperature, country, and crop type.

Approach: Data cleaning → EDA → preprocessing (one‑hot encoding, RobustScaler) → train/test split → model training (Linear, Random Forest, Gradient Boosting) → hyperparameter tuning → evaluation.

. How to Run
●	Install dependencies:  pip install -r requirements.txt
●	Set up Kaggle credentials for kagglehub (a kaggle.json token or the equivalent environment variables).
●	Run  python data_prep.py  to build prepared_data.pkl.
●	Run  python modelling.py  to train the models and generate the results.
The first modelling run is slow because of the grid searches. Every run after that reuses the cached results and is quick.





