import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# 1. Define the Custom Class (MUST match app.py exactly)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        zero_fill_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars']
        for col in zero_fill_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        
        if 'GarageYrBlt' in X.columns and 'YearBuilt' in X.columns:
            X['GarageYrBlt'] = X['GarageYrBlt'].fillna(X['YearBuilt'])
            
        if 'LotFrontage' in X.columns:
            X['LotFrontage'] = X['LotFrontage'].fillna(X['LotFrontage'].median())

        # Feature Creation
        if all(c in X.columns for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
            
        if all(c in X.columns for c in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
            X['TotalBathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath']) + 
                                   X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))
            
        current_year = 2010 
        if 'YearBuilt' in X.columns:
            X['HouseAge'] = current_year - X['YearBuilt']
        if 'YearRemodAdd' in X.columns:
            X['RemodAge'] = current_year - X['YearRemodAdd']
            
        cols_to_drop = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'YearBuilt', 'YearRemodAdd', 'PoolArea']
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
        
        return X

# 2. Load Data
print("Loading data...")
df = pd.read_csv('house.csv') 
X = df.drop(['Id', 'SalePrice'], axis=1)
y = np.log(df['SalePrice']) 

# 3. Define Pipeline
num_features = ['LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'TotalSF', 'TotalBathrooms', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars']
cat_features = ['MSZoning', 'Neighborhood', 'HouseStyle', 'ExterQual', 'KitchenQual']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ],
    remainder='drop' 
)

model_pipeline = Pipeline(steps=[
    ('feature_eng', FeatureEngineer()),
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 4. Train and Save
print("Training model...")
model_pipeline.fit(X, y)
print("Model trained!")

joblib.dump(model_pipeline, 'house_price_model.pkl')
print("Model saved successfully in the virtual environment!")