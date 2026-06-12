import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="Production-ready API serving a Gradient Boosting model with dynamic feature engineering.",
    version="1.0.0"
)

# ----------------------------------------------------------------------
# 1. MODEL LOADING & FALLBACK SECURITY
# ----------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "/app/model/gradient_boosting_churn_model.pkl")

try:
    # Production-da həm model, həm scaler pipeline şəklində yığılmalıdır.
    # Hazırda tək model dump olunduğu üçün fallback məntiqi qururuq.
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Critical Error: Could not load model artifact from {MODEL_PATH}. Details: {str(e)}")

# ----------------------------------------------------------------------
# 2. PYDANTIC INPUT VALIDATION SCHEMA (Minimal & Leakage-Free)
# ----------------------------------------------------------------------
class CustomerDataInput(BaseModel):
    Tenure: int = Field(..., ge=0, le=100, description="Number of months the customer has stayed with the company", example=12)
    MonthlyCharges: float = Field(..., ge=0, description="The amount charged to the customer monthly", example=65.5)
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., example="Month-to-month")
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] = Field(..., example="Electronic check")
    
    # Modelin gözlədiyi digər mühüm kateqoriyalar (Xəta verməməsi üçün defolt təyin edirik)
    Gender: Literal["Male", "Female"] = Field(default="Male")
    SeniorCitizen: Literal[0, 1] = Field(default=0)
    Partner: Literal["Yes", "No"] = Field(default="No")
    Dependents: Literal["Yes", "No"] = Field(default="No")
    PhoneService: Literal["Yes", "No"] = Field(default="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(default="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(default="Fiber optic")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(default="No")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(default="No")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(default="No")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(default="No")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(default="No")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(default="No")
    PaperlessBilling: Literal["Yes", "No"] = Field(default="Yes")


# ----------------------------------------------------------------------
# 3. PREDICTION ENDPOINT
# ----------------------------------------------------------------------
@app.post("/predict", summary="Predict churn risk for a telecom customer")
def predict_churn(input_data: CustomerDataInput):
    try:
        # --- Addm 1: Dynamic Feature Engineering (TotalCharges-ın sızmasını önləyirik) ---
        # Müştəri yenidirsə TotalCharges 0 olur, köhnədirsə tenure * MonthlyCharges ilə bərabərləşdiririk
        calculated_total_charges = 0.0 if input_data.Tenure == 0 else float(input_data.Tenure * input_data.MonthlyCharges)
        
        # ChargesRatio hesablanması (+1 division by zero xətasının qarşısını alır)
        charges_ratio = calculated_total_charges / (input_data.MonthlyCharges + 1.0)
        
        # IsFamily flag-inin hesablanması
        is_family = 1 if (input_data.Partner == "Yes" or input_data.Dependents == "Yes") else 0

        # --- Addım 2: Baza Dataframe-in qurulması ---
        raw_dict = {
            "SeniorCitizen": input_data.SeniorCitizen,
            "Tenure": input_data.Tenure,
            "MonthlyCharges": input_data.MonthlyCharges,
            "TotalCharges": calculated_total_charges,
            "ChargesRatio": charges_ratio,
            "IsFamily": is_family,
            "Gender": input_data.Gender,
            "PhoneService": input_data.PhoneService,
            "MultipleLines": input_data.MultipleLines,
            "InternetService": input_data.InternetService,
            "OnlineSecurity": input_data.OnlineSecurity,
            "OnlineBackup": input_data.OnlineBackup,
            "DeviceProtection": input_data.DeviceProtection,
            "TechSupport": input_data.TechSupport,
            "StreamingTV": input_data.StreamingTV,
            "StreamingMovies": input_data.StreamingMovies,
            "Contract": input_data.Contract,
            "PaperlessBilling": input_data.PaperlessBilling,
            "PaymentMethod": input_data.PaymentMethod
        }
        df_input = pd.DataFrame([raw_dict])

        # --- Addım 3: One-Hot Encoding Alignment (Get Dummies Simulyasiyası) ---
        # Təlim zamanı yaranan tam dummy sütun siyahısı (Notebook-dakı pd.get_dummies nəticəsi)
        # Sənin modelinin tam olaraq gözlədiyi bütün sütunlar bura yazılmalıdır:
        expected_columns = [
            'SeniorCitizen', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesRatio', 'IsFamily',
            'Gender_Male', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
            'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
            'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
            'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year',
            'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ]

        # Giriş datasına dummy tətbiq edirik
        df_encoded = pd.get_dummies(df_input)
        
        # Təlim modelində olmayan sütunları təmizləyirik, əskik olanlara 0 yazırıq (Alignment)
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Sütun ardıcıllığını modelin tam gözlədiyi formaya salırıq
        df_final = df_encoded[expected_columns].astype(float)

        # --- Addım 4: Feature Scaling (Z-score normalizasiyası) ---
        # Sənin notebook-dakı scaler parametrlərini bura fallback olaraq əlavə edirik:
        # (Əgər scaler.pkl yoxdursa, təlim datanın real mean/std ortalamalarını bura yazmalısan)
        num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesRatio']
        
        # Nümunə təlim ortalamaları (Bunu öz real datanın ortalamaları ilə əvəzləyə bilərsən)
        means = {'Tenure': 32.37, 'MonthlyCharges': 64.76, 'TotalCharges': 2279.73, 'ChargesRatio': 32.05}
        stds = {'Tenure': 24.56, 'MonthlyCharges': 30.09, 'TotalCharges': 2266.79, 'ChargesRatio': 24.11}
        
        for col in num_cols:
            df_final[col] = (df_final[col] - means[col]) / (stds[col] + 1e-8)

        # --- Addım 5: Model Inference (Təxminetmə) ---
        prediction = int(model.predict(df_final)[0])
        probability = float(model.predict_proba(df_final)[0][1])
        
        # Risk səviyyəsinin təyin edilməsi
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 2),
            "risk_level": risk_level
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")