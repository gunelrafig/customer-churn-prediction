import logging
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from schemas import CustomerData

logging.basicConfig(
    filename="app_backend.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Telecom Churn Scaled Production API", version="3.9")

try:
    model = joblib.load("gradient_boosting_churn_model.pkl")
    logging.info("Production ML Model successfully initialized.")
except Exception as error:
    logging.critical(f"Failed to load ML model: {str(error)}")
    model = None

# Exact 30 feature columns expected by the model during training
EXPECTED_COLUMNS = [
    'SeniorCitizen', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesRatio', 'IsFamily',
    'Gender_Male', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Global mean and standard deviation values from the Telco dataset (for real-time scaling)
SCALER_MEANS = {
    'Tenure': 32.37,
    'MonthlyCharges': 64.76,
    'TotalCharges': 2283.30,
    'ChargesRatio': 31.95
}
SCALER_STDS = {
    'Tenure': 24.56,
    'MonthlyCharges': 30.09,
    'TotalCharges': 2266.77,
    'ChargesRatio': 23.90
}

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model object is uninitialized.")
    
    try:
        incoming = data.dict()
        
        # 1. Initialize DataFrame with zeros matching the expected structure
        df_final = pd.DataFrame(0, index=[0], columns=EXPECTED_COLUMNS)
        
        # 2. Extract raw numerical features
        raw_tenure = int(incoming["tenure"])
        raw_monthly = float(incoming["monthly_charges"])
        raw_total = float(incoming["total_charges"])
        raw_ratio = raw_total / (raw_monthly + 1)
        
        # 🔥 CRITICAL FIX: Simulate the StandardScaler transformation from the Jupyter Notebook (Z-Score scaling)
        df_final['Tenure'] = (raw_tenure - SCALER_MEANS['Tenure']) / SCALER_STDS['Tenure']
        df_final['MonthlyCharges'] = (raw_monthly - SCALER_MEANS['MonthlyCharges']) / SCALER_STDS['MonthlyCharges']
        df_final['TotalCharges'] = (raw_total - SCALER_MEANS['TotalCharges']) / SCALER_STDS['TotalCharges']
        df_final['ChargesRatio'] = (raw_ratio - SCALER_MEANS['ChargesRatio']) / SCALER_STDS['ChargesRatio']
        
        # 3. Real-time Feature Engineering & Baseline Injections
        df_final['IsFamily'] = 1 if (incoming["partner"] == "Yes" or incoming["dependents"] == "Yes") else 0
        df_final['SeniorCitizen'] = 0
        df_final['PhoneService_Yes'] = 1
        df_final['PaperlessBilling_Yes'] = 1
        
        # 4. Manual One-Hot Dummy Mapping
        # Gender mapping
        if incoming["gender"] == "Male":
            df_final['Gender_Male'] = 1
            
        # Contract mapping (If Month-to-month, both remain 0 to mirror Notebook's drop_first=True approach)
        if incoming["contract"] == "One year":
            df_final['Contract_One year'] = 1
        elif incoming["contract"] == "Two year":
            df_final['Contract_Two year'] = 1
            
        # Internet Service mapping
        if incoming["internet_service"] == "Fiber optic":
            df_final['InternetService_Fiber optic'] = 1
        elif incoming["internet_service"] == "No":
            df_final['InternetService_No'] = 1
            df_final['OnlineSecurity_No internet service'] = 1
            df_final['OnlineBackup_No internet service'] = 1
            df_final['DeviceProtection_No internet service'] = 1
            df_final['TechSupport_No internet service'] = 1
            df_final['StreamingTV_No internet service'] = 1
            df_final['StreamingMovies_No internet service'] = 1
            
        # Payment Method mapping
        if incoming["payment_method"] == "Credit card (automatic)":
            df_final['PaymentMethod_Credit card (automatic)'] = 1
        elif incoming["payment_method"] == "Electronic check":
            df_final['PaymentMethod_Electronic check'] = 1
        elif incoming["payment_method"] == "Mailed check":
            df_final['PaymentMethod_Mailed check'] = 1
            
        # 5. Enforce strict feature order safety alignment
        df_final = df_final[EXPECTED_COLUMNS]
        
        # 6. Execute model inference pipeline
        prediction = model.predict(df_final)[0]
        probability = model.predict_proba(df_final)[0][1]
        
        logging.info(f"Inference complete. Prediction: {prediction}, Churn Prob: {probability:.4f}")
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "status": "success"
        }
        
    except Exception as error:
        logging.error(f"Inference pipeline failure: {str(error)}")
        raise HTTPException(status_code=400, detail=f"Scaling Pipeline Error: {str(error)}")