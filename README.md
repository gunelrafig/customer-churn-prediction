# 📊 Telecom Customer Churn Radar (Enterprise v3.6)

## 🎯 Project Overview & Business Context
Customer attrition (**Customer Churn**) is one of the most critical business metrics directly impacting bottom-line revenue in the telecommunications sector. Financially, the cost of acquiring a new customer is significantly higher than retaining an existing one.

This project is a production-grade, end-to-end Machine Learning solution designed to analyze consumer behavioral data and proactively predict churn probabilities. Engineered using a true **Microservices Architecture**, the system decouples the mathematical prediction engine (Backend API) from the user control panel (Frontend UI) into fully isolated, autonomous environments running lightweight, specialized images managed via **Docker Compose**.

---

## 🏆 Business & Technical Outcomes
By deploying this ecosystem, the following technical and operational capabilities are achieved:
* **Real-Time Risk Stratification:** Customer service representatives or business analysts can input client attributes and receive real-time inference along with an exact churn risk probability scale (e.g., hitting specific benchmarks like 83.0% churn probability).
* **Prescriptive AI Retention Framework:** The system moves beyond binary classification ("Will they leave?"). It analyzes underlying model features (e.g., contract type, friction in payment methods) to dynamically generate **tailored business retention strategies** (e.g., automated payment migration prompts, targeted 15% discount offers).
* **Production-Ready Infrastructure:** The application is entirely decoupled from local OS dependencies, runtime variances, or library conflicts. It can be seamlessly spun up on any cloud infrastructure (AWS, Azure, GCP) supporting Docker with a single command.

---

## 🧰 Tech Stack & Ecosystem
* **Machine Learning Engine:** Built using `Pandas` and `NumPy` for high-throughput matrix manipulations, `Joblib` for model serialization, and **`Scikit-Learn (Gradient Boosting Classifier)`** as the core predictive model due to its robust loss function management on tabular data.
* **Backend Serving API:** Powered by **`FastAPI`** for asynchronous, high-performance execution and native OpenAPI (Swagger) documentation, wrapped with **`Pydantic`** schemas for strict type safety and data validation.
* **Frontend User Interface:** Developed with **`Streamlit`** to provide an intuitive dashboard layout for business users, using the `Requests` library for secure container-to-container communication.
* **Orchestration & DevOps:** Isolated using micro-tuned decentralized `Dockerfiles` per service folder and orchestrated via **`Docker Compose`** to manage virtual network bridging, port allocation, and multi-container state sync.

---

## 📸 Application Preview
Below is the live operational layout of the risk analysis dashboard showing real-time inference and prescriptive playbook strategies:

<img width="1300" height="588" alt="image" src="https://github.com/user-attachments/assets/41cb1982-a581-4b58-8cd7-e66e8a1ced26" />

---

## 🏗️ Architecture & Network Model
The application operates over an isolated, secure internal Docker virtual network bridge where the containers communicate directly using internal service discovery:

```text
[User Browser] 
      │
      ▼
[Streamlit UI (Port 8501)] ──(Docker DNS: http://backend:8000)──► [FastAPI Backend (Port 8000)]
                                                                           │
                                                                           ▼
                                                               [/app/model/gradient_boosting_churn_model.pkl]
🚀 Quick Start (Local Deployment)
Prerequisites
Ensure that Docker Desktop is installed, configured, and actively running (daemon status is active/green) on your host machine.

1. Repository Structure
Verify that your local project root folder matches the following decoupled multi-container layout configuration:

Plaintext
customer-churn-prediction/
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── __init__.py
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── __init__.py
├── model/
│   └── gradient_boosting_churn_model.pkl
├── docker-compose.yml
└── README.md
2. Spinning Up the Cluster
Open your terminal environment, navigate to the project root directory, and launch the multi-container environment while bypassing cached layers to ensure a completely clean deployment:

Bash
# Navigate to the project directory
cd customer-churn-prediction

# Kill any stale dangling containers occupying network ports and rebuild from scratch
docker-compose down
docker-compose up --build
🌐 Verified Service Endpoints
Once the initialization sequence completes successfully, you can access the operational layers via your web browser:

Interactive Frontend UI (Streamlit Dashboard): 🔗 http://localhost:8501

Automated API Documentation (FastAPI Swagger UI): 🔗 http://localhost:8000/docs

🔬 Production Engineering Challenges Resolved
Data Serialization Integrity & Version Pinning: To protect the ensemble tree weight configurations (_gb_losses) from cross-version dependency mutations (which cause silent container failures during runtime), the exact production footprint was strictly locked at scikit-learn==1.3.2.

Microservices Data Alignment: Implemented a robust data validation dictionary layer to normalize naming syntax disparities between raw Streamlit frontend state variables and the uppercase schemas required by the FastAPI/Pydantic inference matrix execution flow (Tenure, MonthlyCharges).

Image Optimization via Decoupled Mappings: Eliminated dependency bloating by splitting the monolithic single-Dockerfile setup into autonomous, service-specific Dockerfiles and requirements.txt manifests. This successfully reduced the Streamlit frontend runtime blueprint by preventing the unnecessary installation of heavy statistical modeling binaries (scikit-learn, pandas).

Multi-Container Network Bridging & Volume Mapping: Resolved the FileNotFoundError loop inside the decoupled backend by establishing a strict Docker Volume Bridge (./model:/app/model) in Compose. This allows the independent FastAPI container to securely load the pickled model binary from a localized absolute path (/app/model/gradient_boosting_churn_model.pkl) without exposing host system file paths or breaking runtime isolation boundaries.
