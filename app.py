import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# SETUP LOGGING
# ==========================================
# This tells Python: "Print any message that is INFO level or higher"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. DEFINE CUSTOM CLASSES
# (Must match the training script exactly)
# ==========================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Implicitly handles missing values for specific cols
        zero_fill_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                          'TotalBsmtSF', 'GarageArea', 'GarageCars']
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

# ==========================================
# 2. LOAD THE MODEL
# ==========================================
# We use st.cache_resource so we only load the model once, not every time the user clicks a button
@st.cache_resource
def load_model():
    model = joblib.load('house_price_model.pkl')
    return model

model = load_model()

# ==========================================
# 3. BUILD THE UI
# ==========================================
st.title("🏡 Ames House Price Prediction")
st.write("Enter the house details below to get an estimated price.")

with st.form("prediction_form"):
    st.header("General Info")
    col1, col2 = st.columns(2)
    
    with col1:
        lot_area = st.number_input("Lot Area (sq ft)", value=5000)
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5)
        year_built = st.number_input("Year Built", 1900, 2025, 2000)
        
        # NEW: Categorical Dropdowns
        st.subheader("Details")
        ms_zoning = st.selectbox("Zoning Classification", 
                                 ['RL', 'RM', 'C (all)', 'FV', 'RH'])
        neighborhood = st.selectbox("Neighborhood", 
                                    ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
                                     'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
                                     'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
                                     'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
                                     'Blueste'])
    
    with col2:
        total_bsmt_sf = st.number_input("Total Basement SF", value=0)
        first_flr_sf = st.number_input("1st Floor SF", value=1000)
        second_flr_sf = st.number_input("2nd Floor SF", value=0)
        full_bath = st.number_input("Full Bathrooms", 0, 5, 1)
        
        # NEW: Categorical Dropdowns
        st.subheader("Finishes")
        house_style = st.selectbox("House Style", 
                                   ['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'])
        exter_qual = st.selectbox("Exterior Quality", 
                                  ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
        kitchen_qual = st.selectbox("Kitchen Quality", 
                                    ['Ex', 'Gd', 'TA', 'Fa', 'Po'])

    # Submit Button
    submit_btn = st.form_submit_button("Predict Price")

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if submit_btn:
    # 1. Dictionary of inputs
    input_data = {
        'LotArea': [lot_area],
        'OverallQual': [overall_qual],
        'OverallCond': [overall_cond],
        'YearBuilt': [year_built],
        'TotalBsmtSF': [total_bsmt_sf],
        '1stFlrSF': [first_flr_sf],
        '2ndFlrSF': [second_flr_sf],
        'FullBath': [full_bath],
        
        # Now mapped to the user selection
        'MSZoning': [ms_zoning],
        'Neighborhood': [neighborhood],
        'HouseStyle': [house_style],
        'ExterQual': [exter_qual],
        'KitchenQual': [kitchen_qual],

        # Still handling defaults for things we didn't expose to keep UI clean
        'MasVnrArea': [0],
        'BsmtFinSF1': [0],
        'BedroomAbvGr': [3],
        'TotRmsAbvGrd': [6],
        'GarageCars': [1],
        'HalfBath': [0],
        'BsmtFullBath': [0],
        'BsmtHalfBath': [0],
        'YearRemodAdd': [year_built], 
        'GarageYrBlt': [year_built]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # 2. Predict
    prediction_log = model.predict(input_df)
    prediction = np.exp(prediction_log[0])
    
    # 3. LOG THE EVENT (This is the Monitoring part)
    # We record the inputs and the output.
    logger.info(f"PREDICTION EVENT: Inputs=[Area:{lot_area}, Qual:{overall_qual}, Yr:{year_built}], Output=${prediction:,.2f}")
    
    # 4. Display the result
    st.success(f"Estimated Price: ${prediction:,.2f}")