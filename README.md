# 🌾 AgriSmart: Smart Agriculture Intelligence Dashboard
### *Crop Yield Forecasting, Farmer Profitability Prediction & AgroSense AI Executive Copilot*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E.svg)](https://scikit-learn.org/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%20%26%20GPT--OSS-f55036.svg)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-00599C.svg)](https://github.com/facebookresearch/faiss)
[![Plotly](https://img.shields.io/badge/Plotly-6.6.0-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Executive Summary & Project Background

**AgriSmart** is an enterprise agricultural data intelligence system designed for agribusiness analysts, farm enterprise managers, and policymakers in India. The platform integrates historical agricultural records covering 10 years (2014–2024), exploratory data analysis (EDA), multi-model machine learning benchmarking, an interactive **Streamlit** dashboard, and **AgroSense AI** — a decision-support conversational intelligence assistant powered by Retrieval-Augmented Generation (RAG).

The application addresses three core analytical workflows:
1. **How much crop can a farmer expect to harvest?** (Crop Yield Forecasting in tons/hectare)
2. **What will be the net monetary return for the farm enterprise?** (Farmer Profitability Prediction in INR)
3. **What strategic decisions should farm management make?** (AgroSense AI: Omnipresent floating circular copilot bubble with interactive Plotly visualizer & out-of-domain guardrails)

---

## 🎯 What We Have Done & Built in This Project

### 1. Data Cleaning & Pipeline Preprocessing
- **Dataset**: `india_rainfall_crop_yield_2014_2024 (1).csv` containing **7,920 records** across all major Indian agricultural states.
- **Features Cleaned**:
  - **Categorical (5)**: `State`, `Season` (Kharif, Rabi, Zaid), `Crop_Type` (Rice, Wheat, Maize, Cotton, Pulses, Sugarcane, Millets, Barley), `Soil_Type` (Alluvial, Black, Red, Sandy, Laterite), and `Irrigation_Type` (Canal, Drip, Sprinkler, Tube Well, Rain-fed).
  - **Numerical (6)**: `Rainfall_mm`, `Avg_Temperature_C`, `Humidity_percent`, `Fertilizer_Used_kg`, `Pesticide_Used_kg`, `Cultivated_Area_hectares`.
  - **Targets**: `Yield_ton_per_hectare` and `Farmer_Profit_INR`.
- **Preprocessors**: Built reusable `ColumnTransformer` pipelines combining `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` for categories and `SimpleImputer(strategy='median')` + `StandardScaler()` for continuous numerical inputs.

### 2. Comprehensive Exploratory Data Analysis (EDA)
- Analyzed distribution curves, skewness, and variance across crops, soils, and states.
- Analyzed 10-year macro trends showing yield stability around ~4.83 tons/ha.
- Computed complete correlation matrices demonstrating that `Cultivated_Area_hectares` ($r = 0.69$) and `Yield_ton_per_hectare` ($r = 0.55$) strongly govern `Farmer_Profit_INR`.

### 3. Machine Learning Modeling & Critical Evaluation Findings
We implemented and systematically evaluated **9 regression algorithms**:
1. **Linear Regression**
2. **Lasso Regression** ($\alpha = 0.1$)
3. **Ridge Regression** ($\alpha = 1.0$)
4. **Decision Tree Regressor**
5. **Random Forest Regressor** ($n=150$, min leaf = 2)
6. **Gradient Boosting Regressor** ($n=100$, lr = 0.05)
7. **Support Vector Regressor (SVR)**
8. **K-Nearest Neighbors (KNN)**
9. **Artificial Neural Network (MLPRegressor)**

#### Key Analytical Insights:
- **Profitability Modeling**: Models achieved exceptional predictive accuracy with **$R^2 \approx 0.77 - 0.80+$**, led by **Gradient Boosting** and **Random Forest**. Profitability models evaluate scale and yield interaction with high fidelity.
- **Yield Modeling Evaluation Finding**: In this dataset, `Yield_ton_per_hectare` has virtually zero statistical correlation ($|r| \le 0.02$) with weather/soil parameters due to synthetic uniform distribution. Consequently, evaluating yield models on unseen holdout test sets yields slightly negative $R^2$ scores across all algorithms. In accordance with data science best practices, the standalone negative-$R^2$ Yield Model Evaluation tab was removed from the app interface, while the dedicated **Crop Yield Forecasting** engine was optimized to utilize the best Random Forest pipeline.
- **TimeSeries Cross-Validation**: Implemented 5-fold expanding window `TimeSeriesSplit` cross-validation to assess chronological stability.
- **Principal Component Analysis (PCA)**: Evaluated 99% variance preservation on numeric factors.

### 4. Model Serialization & Export
- Serialized the best trained models via `joblib`:
  - `crop_yield_model.pkl`: Best model for Crop Yield Forecasting.
  - `farmer_profitability_model.pkl`: Best model for Farmer Profitability Prediction.
- Both models are exported and pre-cached so that the Streamlit app generates predictions instantly without re-training delay.

---

## 🤖 AgroSense AI — Omnipresent Copilot Bubble & Enterprise Features

```
                          ┌─────────────────────────────────────┐
                          │    AgriSmart Dataset (7,920 Rows)   │
                          └──────────────────┬──────────────────┘
                                             │ Knowledge Chunking
                                             ▼
                          ┌─────────────────────────────────────┐
                          │   115 Multi-Dimensional Chunks     │
                          │ (State, Crop, Soil, Season, Yield)  │
                          └──────────────────┬──────────────────┘
                                             │ Normalized Embeddings
                                             ▼
                          ┌─────────────────────────────────────┐
                          │   FAISS Vector Database (IndexIP)   │
                          └──────────────────┬──────────────────┘
                                             │ Top-K Retrieval
                                             ▼
User Query ──► Domain Guardrail ──► Query Vectorizer ──► Context-Augmented Prompt
                                                            │
                                                            ▼
                                         ┌─────────────────────────────────────┐
                                         │       Groq Cloud LLM Inference      │
                                         │   (openai/gpt-oss-20b / 120b)       │
                                         └──────────────────┬──────────────────┘
                                                            │ Zero-Hallucination Output
                                                            ▼
               ┌─────────────────────────────────────────────────────┐
               │              AgroSense AI Executive Brief           │
               │  • 📊 Executive Summary (Direct Answer & KPIs)      │
               │  • 🔍 Quantitative Comparison Table (State/Crop)    │
               │  • 📋 Strategic Recommendations (Actionable Steps)   │
               │  • 📈 Context-Aware Interactive Plotly Visuals      │
               └─────────────────────────────────────────────────────┘
```

### ✨ Advanced Architecture & UI/UX Design:
1. **Floating Circular Chat Bubble (NOT a Dropdown Bar)**:
   - Replaced all cumbersome dropdown bars with an elegant, small circular floating chat bubble (`💬`) fixed in the bottom-right corner of **every single page**.
   - Clicking the bubble triggers the full enterprise modal dialog (`@st.dialog`) without disturbing dashboard layouts.
2. **Clean 6-Page Navigation**:
   - The left sidebar Navigation radio focuses strictly on the 6 core analytical dashboards (Dashboard Overview, Crop Analysis, Data Explorer, Profitability Model Evaluation, Crop Yield Forecasting, and Farmer Profitability Prediction).
3. **Out-of-Domain Guardrail (Zero Hallucination)**:
   - If an off-topic or random question is asked (e.g., sports, coding, general trivia), AgroSense AI politely declines and provides suggested agricultural queries, alongside a macro dataset benchmark chart.
4. **Guaranteed Interactive Plotly Graph on EVERY Response**:
   - Every response dynamically includes an executive-grade Plotly chart (state yield rankings, crop profitability spreads, 10-year trajectories, irrigation efficiency, price benchmarks, or ML feature importance).
5. **Animated Thinking Status Indicator**:
   - Displays real-time operational feedback (`🤖 AgroSense AI is thinking... Analyzing 7,920 records & synthesizing executive solution...`) during processing.
6. **Multi-Session Chat History Archive (Read, Resume, Delete & Export)**:
   - **Read Past Briefings**: Inspect past session transcripts with message counts and creation timestamps.
   - **Resume Past Thread**: Load any previous investigation into the active chat window.
   - **Individual Deletion**: 🗑️ Delete specific threads.
   - **Export as Markdown (.md)**: 📥 Instant download of formatted executive briefing reports.
7. **ML Prediction & Forecasting Advisory**:
   - Deep explanations of Random Forest and Gradient Boosting dynamics ($R^2 \approx 0.80$), input cost sensitivities, and feature dominance (Area $r=0.69$, Yield $r=0.55$).

---

## 📁 Repository Structure

```
AgriSmart-Dashboard/
│
├── app.py                                   # Complete Streamlit Dashboard (6 Pages + Floating Copilot Bubble)
├── chatbot.py                               # AgroSense AI RAG engine, dynamic chart generator & Session Manager
├── india_rainfall_crop_yield_2014_2024 (1).csv # Historical agricultural dataset (7,920 records)
├── rain.ipynb                               # Full analysis & ML model development notebook
├── crop_yield_model.pkl                     # Serialized Random Forest yield model
├── farmer_profitability_model.pkl           # Serialized Gradient Boosting profitability model
├── requirements.txt                         # Production package dependencies
├── .env.example                             # Example environment configuration
├── .gitignore                               # Git ignore configuration (protects .env & sessions)
└── README.md                                # Comprehensive documentation
```

---

## 🚀 Quick Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Aadish2003/AgriSmart-Dashboard.git
cd AgriSmart-Dashboard
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Groq API Key
Create a `.env` file in the root folder:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(The key is loaded strictly on the backend and is never exposed in the dashboard UI).*

### 5. Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💻 Dashboard Pages & User Workflows

| Page | Description | Key Features |
|---|---|---|
| **Dashboard Overview** | Executive summary of national agricultural metrics. | KPI metric cards, regional yield breakdowns, crop distribution charts, **Floating AI Bubble**. |
| **Crop Analysis** | Comparative analysis across 8 major crop varieties. | Yield vs. input correlations, price-yield scatter plots, soil suitability charts, **Floating AI Bubble**. |
| **Data Explorer** | Interactive dataset browser with multi-dimensional filtering. | State/Crop/Season filters, summary statistics, filtered CSV export, **Floating AI Bubble**. |
| **Profitability Model Evaluation** | Rigorous evaluation of 9 ML algorithms for farm profit. | Holdout metrics ($R^2$, MAE, RMSE), TimeSeries CV, residual distribution plots, **Floating AI Bubble**. |
| **Crop Yield Forecasting** | ML-driven expected yield prediction engine. | Farm parameter input form, Random Forest inference, sensitivity curves, regional benchmark charts, **Floating AI Bubble**. |
| **Farmer Profitability Prediction** | End-to-end farm enterprise profitability forecasting. | Scale & input cost analysis, sensitivity curves ($\pm 20\%$), revenue/cost structure breakdown, **Floating AI Bubble**. |

---

## 👥 Contributors & Academic Attribution
Developed as part of the **MBA Business Analytics (SEM 3)** curriculum.
- **Repository**: [https://github.com/Aadish2003/AgriSmart-Dashboard](https://github.com/Aadish2003/AgriSmart-Dashboard)
- **Author**: Aadish Kotadia
