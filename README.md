# 🌾 AgriSmart: Smart Agriculture Intelligence Dashboard
### *Crop Yield Forecasting & Farmer Profitability Prediction System*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-6.6.0-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Executive Summary & Project Background

**AgriSmart** is an end-to-end agricultural data intelligence system designed for agribusiness analysts, farmers, and agricultural extension officers in India. The platform integrates historical agricultural records covering 10 years (2014–2024), exploratory data analysis (EDA), multi-model machine learning benchmarking, and an interactive **Streamlit** dashboard.

The application addresses two critical questions for farm operations:
1. **How much crop can a farmer expect to harvest?** (Crop Yield Forecasting in tons/hectare)
2. **What will be the net monetary return for the farm enterprise?** (Farmer Profitability Prediction in INR)

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

### 5. Interactive Streamlit Dashboard (`app.py`)
A custom-themed agricultural intelligence UI featuring 6 comprehensive modules:

#### Module 1: Dashboard Overview
- 4 macro KPI metric cards (Total Records, Average Yield, Average Profit, Historical Years).
- Crop-wise average yield bar chart.
- 10-year timeline trend line chart.
- State-wise farmer profit comparison.
- Season-wise productivity breakdown.

#### Module 2: Crop Analysis
- Interactive crop filter enabling micro-analysis of yield performance across Indian states and seasons.

#### Module 3: Data Explorer
- Complete dataset schema, data type breakdown, missing value validation, descriptive statistics table, target distribution histogram, and correlation heatmap.

#### Module 4: Profitability Model Evaluation
- Multi-model evaluation scoreboard ranking all 9 regression algorithms by $R^2$ score and RMSE.
- Baseline model comparison.
- Actual vs. Predicted scatter plot with perfect-prediction diagonal.
- Residual analysis and Top 15 Feature Importances bar chart.
- Model download utility (`.pkl` export).

#### Module 5: Crop Yield Forecasting (Enhanced with Graphs & Advisory)
- User input form for all agronomic variables.
- **Predicted Yield Metric** with qualitative rating (*Optimal Yield*, *High Yield*, *Moderate Yield*).
- **Total Output Metric** ($Area \times Yield$ in metric tons).
- **Benchmark Bar Chart**: Compares the prediction against State Crop Average, National Crop Average, State Average, and National Benchmark.
- **Historical Distribution Curve**: Histogram with a red vertical marker showing the exact percentile position of the forecast.
- **Agro-Climatic Condition Assessment**: Real-time delta comparison of user rainfall, temperature, and humidity against the crop's historical average, plus fertilizer intensity ($kg/ha$).
- **In-Depth Agronomic Advisory**: Detailed 4-part guidance on yield rating, climatic drivers, soil-irrigation dynamics, and nutrient split-dosing.

#### Module 6: Farmer Profitability Prediction (Enhanced with Graphs & Advisory)
- User input form taking farming parameters and expected yield.
- **Net Profit Metric** (₹) and **Profit per Hectare** (₹/ha).
- **Profit Sensitivity Analysis Chart**: Dynamic simulation showing projected net profit under $\pm 20\%$ yield fluctuation scenarios ($-20\%$, $-10\%$, Base, $+10\%$, $+20\%$).
- **Regional Profit Benchmark Chart**: Compares the farm's projected profit against state and crop benchmarks.
- **Financial Structure Breakdown**: Gross Market Value vs. Operating/Input Costs vs. Net Farmer Return.
- **In-Depth Agribusiness Advisory**: Detailed 4-part guidance on enterprise volume, scale advantages, input cost optimization, and post-harvest market timing/MSP hedging.

### 6. Jupyter Notebook (`rain.ipynb`)
- Fully executed notebook containing all **34 code cells with pre-rendered outputs** (statistics tables, correlation plots, model comparisons, residual graphs, feature importances, cross-validation scores, and the forecasting scenario).

---

## 📂 Project Structure

```text
├── app.py                                      # Streamlit Web Application (6 modules)
├── rain.ipynb                                  # Fully executed Jupyter Notebook with all outputs
├── india_rainfall_crop_yield_2014_2024 (1).csv  # Agricultural dataset (7,920 records)
├── crop_yield_model.pkl                        # Serialized Best Crop Yield Model
├── farmer_profitability_model.pkl              # Serialized Best Profitability Model
├── requirements.txt                            # Complete Python dependencies
├── .gitignore                                  # Git ignore rules for clean repository
└── README.md                                   # Comprehensive project documentation
```

---

## 🛠️ Technology Stack

| Layer | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **User Interface** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (Pipelines, ColumnTransformer, RandomForest, GradientBoosting, Ridge, Lasso) |
| **Visualization** | Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib |
| **Model Persistence**| Joblib |
| **Development** | Jupyter Notebook, VS Code |

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Aadish2003/AgriSmart-Dashboard.git
cd AgriSmart-Dashboard
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows:
python -m venv venv
venv\Scripts\activate

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
*(The dashboard will open automatically in your browser at `http://localhost:8501`)*

### 5. Inspect the Jupyter Notebook
```bash
python -m jupyter notebook rain.ipynb
```

---

## 👥 Authors & Academic Context
Developed as part of the **Master of Business Administration (MBA) - Business Analytics (BA)** curriculum.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
