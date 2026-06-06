import streamlit as st
import requests

st.set_page_config(page_title="Churn Radar Pro v3.6", layout="wide")

st.title("📡 Telecom Customer Churn Radar (Enterprise v3.6)")
st.subheader("Production Pipeline Powered by Top-10 High-Impact Features")

st.sidebar.header("🎯 Highly Influential Customer Metrics")

# Giriş Slayderləri və Seçimlər
tenure = st.sidebar.slider("Tenure (Months with company)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 10.0, 150.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 10.0, 10000.0, 840.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])

if st.sidebar.button("Run Risk Inference Pipeline"):
    # Schemas.py-dakı tam kiçik hərflərlə yazılmış struktura uyğun payload
    payload = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "gender": gender,
        "partner": partner,
        "dependents": dependents
    }
    
    try:
        with st.spinner("Processing features and syncing with ML pipeline..."):
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()
        
        if response.status_code == 200:
            prob = result["churn_probability"]
            st.metric(label="Calculated Customer Churn Risk", value=f"{prob * 100:.1f}%")
            
            if result["churn_prediction"] == 1:
                st.error("🚨 CRITICAL WARNING: High risk profile detected!")
                st.write("---")
                st.subheader("💡 AI-Powered Prescriptive Retention Strategies:")
                
                if contract == "Month-to-month":
                    st.warning("⚠️ Contract Risk: Customer is on a Month-to-month plan. Offer a long-term contract package with a 15% discount.")
                if internet_service == "Fiber optic":
                    st.info("⚡ Fiber Optic Alert: High churn observed in fiber customers. Schedule an immediate proactive technical satisfaction review.")
                if payment_method == "Electronic check":
                    st.warning("💳 Payment Friction: Electronic check payments are heavily correlated with churn. Offer a one-time $5 bill credit if they set up Credit Card or Bank Auto-Pay.")
                if partner == "No" and dependents == "No":
                    st.info("👤 Single Account Volatility: Customer lacks family anchors. Recommend multi-line bundle options.")
            else:
                st.success("✅ Stable Account Profile: Low churn risk detected.")
        else:
            st.error(f"API Processing Failure: {result['detail']}")
            
    except Exception as error:
        st.error(f"Failed to connect to Backend API. Error: {str(error)}")