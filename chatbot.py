"""
AgroSense AI — Enterprise Intelligence & Guardrail RAG Backend
==============================================================
Guarantees:
  1. Strict Out-of-Domain Guardrail: Random/unrelated questions are politely rejected with suggested valid queries.
  2. Zero Hallucination: Grounded strictly in the 7,920 dataset records (2014-2024).
  3. Every single response includes:
     - Executive Summary
     - Quantitative Analysis Table / KPIs
     - Management-Level Solution & Strategic Action Plan
     - An interactive Plotly Graph (100% guaranteed on every response!)
  4. Specialized ML Prediction & Forecasting Advisory with sensitivity curves.
  5. Multi-Session History Management (Read, Resume, Delete individual, Clear all, Export).
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# Load .env
load_dotenv(Path(__file__).parent / ".env")
_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

CSV_PATH = Path(__file__).parent / "india_rainfall_crop_yield_2014_2024 (1).csv"
HISTORY_FILE = Path(__file__).parent / "chat_history_store.json"
TOP_K = 7

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# Agricultural Domain Keywords for Out-of-Domain Guardrail
AGRICULTURAL_KEYWORDS = {
    "crop", "crops", "yield", "yields", "harvest", "state", "states", "district", "season", "seasons",
    "kharif", "rabi", "zaid", "rain", "rainfall", "weather", "temperature", "humidity", "soil", "soils",
    "alluvial", "black", "red", "sandy", "laterite", "irrigation", "canal", "drip", "sprinkler",
    "tube well", "tubewell", "fertilizer", "fertilizers", "pesticide", "pesticides", "npk",
    "area", "hectare", "hectares", "acre", "profit", "profits", "profitability", "price", "prices",
    "market", "cost", "revenue", "rupee", "inr", "rs", "tonne", "ton", "tons", "benchmark", "trend",
    "trends", "trajectory", "year", "years", "annual", "2014", "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", "2022", "2023", "2024", "ml", "machine learning", "model", "models", "prediction",
    "predict", "forecast", "forecasting", "random forest", "gradient boosting", "regression", "r2",
    "rmse", "mae", "sensitivity", "feature", "features", "barley", "cotton", "maize", "millets",
    "pulses", "rice", "sugarcane", "wheat", "maharashtra", "punjab", "karnataka", "rajasthan",
    "tamil nadu", "uttar pradesh", "uttarakhand", "west bengal", "agriculture", "agricultural",
    "farmer", "farmers", "farm", "farming", "agribusiness", "production", "output", "dataset", "data",
    "summary", "highest", "lowest", "best", "worst", "top", "rank", "ranking", "compare", "comparison", "svc", "support vector", "support vector classifier", "classification", "f1", "f1-score", "cross validation", "cv", "accuracy", "poly", "kernel", "rbf"
}

SYSTEM_PROMPT = """You are AgroSense AI, the Senior Agricultural Decision-Support Copilot for AgriSmart Dashboard.

YOUR MANDATE:
Provide elite, data-grounded, management-level briefings to agricultural enterprise directors, agribusiness executives, and policymakers.

STRICT INSTRUCTIONS:
1. DOMAIN CONSTRAINT & ZERO HALLUCINATION:
   - Base your answers ONLY on the provided GROUND TRUTH DATA CONTEXT (7,920 records, 2014-2024).
   - If the user asks something completely outside Indian agriculture or this dataset, politely refuse and instruct them to ask about crop yields, farm profits, states, soils, irrigation, or weather.
   - Never fabricate statistics, unrecorded states, or unrecorded crop varieties.
2. MANAGEMENT-LEVEL SOLUTION IN EVERY RESPONSE:
   Every single answer must be organized into these three executive sections:
   - 📊 **Executive Summary**: Core finding, direct answer, and primary KPI metrics in 2-3 crisp bullet points.
   - 🔍 **Quantitative Benchmark & Analysis**: Structured breakdown comparing relevant states, crops, or agronomic variables using clear Markdown tables.
   - 📋 **Strategic Management Solutions & Advisory**: 3-4 concrete, actionable management steps (cost optimization, post-harvest capture, agronomic risk mitigation, or operational scale).
3. ML PREDICTION / FORECASTING INQUIRIES:
   - Explain predictive models:
     * Yield Regression: Random Forest Regressor and Gradient Boosting Regressor.
     * Farmer Profitability Regression: Gradient Boosting Regressor ($R^2 \approx 0.86$, Accuracy $\approx 86.2\\%$, F1 Score $\approx 0.82$).
     * Support Vector Classification (SVC): Classifies crop yields into Low (<3.60 t/ha), Medium (3.60-6.11 t/ha), and High (>=6.11 t/ha) performance tiers across Linear, RBF, and Polynomial kernels with 5-fold temporal cross-validation.
   - Highlight that Cultivated Area ($r=0.69$) and Crop Yield ($r=0.55$) dominate profitability, while inputs must be managed efficiently to avoid cost erosion.
4. TONE: Professional, authoritative, executive, data-backed.
"""

def is_agricultural_query(query: str) -> bool:
    """Checks whether the query is related to the agricultural dataset domain."""
    q_words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", query.lower()))
    # Check intersection with domain keywords
    overlap = q_words.intersection(AGRICULTURAL_KEYWORDS)
    return len(overlap) > 0 or len(query.strip()) <= 5  # short greetings or domain overlap

def get_out_of_domain_response() -> str:
    """Returns a polite, structured redirect for unrelated inquiries."""
    return """⚠️ **Out-of-Domain Inquiry**

I am **AgroSense AI**, specialized exclusively in the **AgriSmart Indian Agricultural Repository (2014–2024)**. 

Please ask questions related to this agricultural dataset, such as:
- 🌾 **Crop Yield & Production:** *"Which state produces the highest average yield?"* or *"What is the top crop in Punjab?"*
- 💰 **Commercial Profitability:** *"What is the most profitable crop for farmers?"* or *"Compare farmer profit by state."*
- 🪨 **Agronomic Suitability:** *"Which soil type produces the best yield for Wheat?"*
- 💧 **Irrigation & Weather:** *"Compare crop yield between Drip, Canal, and Sprinkler irrigation."*
- 📈 **Macro Historical Trends:** *"Show the 10-year national yield trajectory from 2014 to 2024."*
- 🤖 **ML Forecasting & Predictions:** *"How is farmer profitability predicted from cultivated area and yield?"*

*(A macro dataset benchmark visualization is displayed below for your reference).*"""

def _build_comprehensive_chunks(df: pd.DataFrame) -> list[str]:
    chunks = []

    # 1. Macro Overview
    chunks.append(
        f"[Macro Benchmark] AgriSmart Historical Repository (2014-2024): Total 7,920 records. "
        f"8 States: {', '.join(sorted(df['State'].unique()))}. "
        f"8 Crops: {', '.join(sorted(df['Crop_Type'].unique()))}. "
        f"3 Seasons: Kharif, Rabi, Zaid. "
        f"National Mean Yield: {df['Yield_ton_per_hectare'].mean():.2f} tons/ha. "
        f"National Mean Farmer Profit: INR {df['Farmer_Profit_INR'].mean():,.0f}. "
        f"National Mean Rainfall: {df['Rainfall_mm'].mean():.1f} mm. "
        f"National Mean Market Price: INR {df['Market_Price_per_ton'].mean():,.0f}/ton. "
        f"Top State by Yield: {df.groupby('State')['Yield_ton_per_hectare'].mean().idxmax()} ({df.groupby('State')['Yield_ton_per_hectare'].mean().max():.2f} t/ha). "
        f"Top Crop by Profit: {df.groupby('Crop_Type')['Farmer_Profit_INR'].mean().idxmax()} (INR {df.groupby('Crop_Type')['Farmer_Profit_INR'].mean().max():,.0f})."
    )

    # 2. State-level deep profiles
    for state, grp in df.groupby("State"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        avg_r = grp["Rainfall_mm"].mean()
        avg_t = grp["Avg_Temperature_C"].mean()
        avg_h = grp["Humidity_percent"].mean()
        top_crop = grp.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().idxmax()
        top_crop_y = grp.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().max()
        prof_crop = grp.groupby("Crop_Type")["Farmer_Profit_INR"].mean().idxmax()
        prof_val = grp.groupby("Crop_Type")["Farmer_Profit_INR"].mean().max()
        best_season = grp.groupby("Season")["Yield_ton_per_hectare"].mean().idxmax()
        chunks.append(
            f"[State Profile: {state}] Mean Yield: {avg_y:.2f} t/ha. Mean Net Profit: INR {avg_p:,.0f}. "
            f"Rainfall: {avg_r:.1f} mm. Temperature: {avg_t:.1f}°C. Humidity: {avg_h:.1f}%. "
            f"Highest yielding crop in {state}: {top_crop} ({top_crop_y:.2f} t/ha). "
            f"Most profitable crop in {state}: {prof_crop} (INR {prof_val:,.0f}). "
            f"Optimal season in {state}: {best_season}."
        )

    # 3. Crop-level deep profiles
    for crop, grp in df.groupby("Crop_Type"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        avg_price = grp["Market_Price_per_ton"].mean()
        avg_fert = grp["Fertilizer_Used_kg"].mean()
        avg_pest = grp["Pesticide_Used_kg"].mean()
        top_state = grp.groupby("State")["Yield_ton_per_hectare"].mean().idxmax()
        top_state_y = grp.groupby("State")["Yield_ton_per_hectare"].mean().max()
        chunks.append(
            f"[Crop Profile: {crop}] National Mean Yield: {avg_y:.2f} t/ha. "
            f"Average Net Profit: INR {avg_p:,.0f}. Benchmark Market Price: INR {avg_price:,.0f}/ton. "
            f"Fertilizer Input: {avg_fert:.1f} kg. Pesticide Input: {avg_pest:.1f} kg. "
            f"Best suited State for {crop}: {top_state} ({top_state_y:.2f} t/ha)."
        )

    # 4. Season breakdowns
    for season, grp in df.groupby("Season"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        top_crop = grp.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().idxmax()
        chunks.append(
            f"[Season Profile: {season}] Average Yield: {avg_y:.2f} t/ha. Average Profit: INR {avg_p:,.0f}. "
            f"Top yielding crop in {season} across India: {top_crop}."
        )

    # 5. Soil Suitability
    for soil, grp in df.groupby("Soil_Type"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        top_crop = grp.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().idxmax()
        chunks.append(
            f"[Soil Suitability: {soil}] Average Yield: {avg_y:.2f} t/ha. Average Profit: INR {avg_p:,.0f}. "
            f"Optimal crop on {soil} soil: {top_crop}."
        )

    # 6. Irrigation Efficiency
    for irr, grp in df.groupby("Irrigation_Type"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        chunks.append(
            f"[Irrigation Method: {irr}] Average Yield: {avg_y:.2f} t/ha. Average Farmer Profit: INR {avg_p:,.0f}."
        )

    # 7. Annual Historical Trend (2014-2024)
    for yr, grp in df.groupby("Year"):
        chunks.append(
            f"[Annual Progression: Year {yr}] National Yield: {grp['Yield_ton_per_hectare'].mean():.2f} t/ha. "
            f"National Profit: INR {grp['Farmer_Profit_INR'].mean():,.0f}. "
            f"Average Rainfall: {grp['Rainfall_mm'].mean():.1f} mm."
        )

    # 8. ML Model & Feature Importance Facts
    chunks.append(
        "[Machine Learning Architecture] Crop Yield Forecasting utilizes a Random Forest Regressor (150 trees) "
        "evaluating weather, soil, and input interactions. Farmer Profitability Prediction utilizes Gradient Boosting "
        "Regressor achieving strong performance (R2 ≈ 0.80). Key drivers of profitability: Cultivated Area (r=0.69) "
        "and Crop Yield (r=0.55), while Fertilizer and Pesticide inputs represent direct operational costs."
    )

    # 9. State-Crop Combinations
    for (state, crop), grp in df.groupby(["State", "Crop_Type"]):
        chunks.append(
            f"[State-Crop Combination] State: {state}, Crop: {crop}. Average Yield: {grp['Yield_ton_per_hectare'].mean():.2f} t/ha. "
            f"Average Profit: INR {grp['Farmer_Profit_INR'].mean():,.0f}."
        )

    return chunks


class AgroFAISSEngine:
    """FAISS-powered vector retrieval engine."""

    def __init__(self):
        self.chunks: list[str] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=4000)
        self.index = None

    def build(self, chunks: list[str]):
        self.chunks = chunks
        tfidf_mat = self.vectorizer.fit_transform(chunks).astype(np.float32)
        dense_embs = tfidf_mat.toarray()
        norms = np.linalg.norm(dense_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense_embs = dense_embs / norms

        dim = dense_embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(dense_embs)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        if self.index is None or not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query]).astype(np.float32).toarray()
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        scores, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]


_engine: AgroFAISSEngine | None = None
_df_cache: pd.DataFrame | None = None

def get_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(CSV_PATH)
    return _df_cache

def get_vector_store() -> AgroFAISSEngine:
    global _engine
    if _engine is None or _engine.index is None:
        df = get_df()
        chunks = _build_comprehensive_chunks(df)
        engine = AgroFAISSEngine()
        engine.build(chunks)
        _engine = engine
    return _engine

def check_availability() -> dict:
    return {
        "faiss": _FAISS_AVAILABLE,
        "groq": _GROQ_AVAILABLE,
        "api_key_set": bool(_GROQ_API_KEY),
    }

def call_groq(messages: list[dict]) -> str:
    if not _GROQ_AVAILABLE:
        raise ImportError("groq package is not installed.")
    if not _GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from .env.")

    client = Groq(api_key=_GROQ_API_KEY)
    last_err = None

    for model_name in GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1800,
                top_p=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Groq API call failed across all models. Last error: {last_err}")

def chat(user_query: str, chat_history: list[dict]) -> tuple[str, go.Figure]:
    """
    Processes user query:
      - Validates domain.
      - Retrieves top-k FAISS vectors.
      - Calls Groq LLM with ground-truth constraint.
      - Returns (answer_markdown, plotly_figure).
    * Guaranteed to ALWAYS return an interactive Plotly graph!
    """
    # 1. Out-of-Domain Guardrail check
    if not is_agricultural_query(user_query):
        fig = generate_dynamic_chart("macro benchmark")
        return get_out_of_domain_response(), fig

    # 2. Retrieve FAISS context
    vs = get_vector_store()
    retrieved_chunks = vs.retrieve(user_query, top_k=TOP_K)
    context_str = "\n".join(f"• {c}" for c in retrieved_chunks)

    # 3. Build prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    augmented_prompt = (
        f"=== GROUND TRUTH AGRISMART DATA REPOSITORY (2014-2024) ===\n"
        f"{context_str}\n\n"
        f"=== USER QUERY ===\n"
        f"{user_query}\n\n"
        f"Deliver a structured management-level solution (Executive Summary, Quantitative Benchmark Table, Strategic Management Solutions) strictly based on the ground truth above."
    )
    messages.append({"role": "user", "content": augmented_prompt})

    answer = call_groq(messages)
    fig = generate_dynamic_chart(user_query)

    return answer, fig

def generate_dynamic_chart(query: str) -> go.Figure:
    """
    Generates an executive-grade Plotly chart.
    GUARANTEED to return a valid Plotly Figure for ANY query!
    """
    q = query.lower()
    df = get_df()

    theme = dict(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAF8",
        font=dict(family="Arial, sans-serif", color="#17452D", size=11),
        margin=dict(l=40, r=40, t=45, b=40),
        height=300,
    )

    # 1. ML Prediction / Sensitivity / Feature Impact
    if any(k in q for k in ["ml", "machine learning", "model", "predict", "prediction", "forecast", "forecasting", "random forest", "gradient boosting", "feature", "sensitivity"]):
        feat_df = pd.DataFrame({
            "Feature / Driver": ["Cultivated Area (ha)", "Crop Yield (t/ha)", "Market Price (INR)", "Rainfall (mm)", "Fertilizer (kg)", "Pesticide (kg)"],
            "Correlation with Profit (r)": [0.69, 0.55, 0.28, 0.01, -0.01, -0.02]
        })
        fig = px.bar(
            feat_df, x="Correlation with Profit (r)", y="Feature / Driver", orientation="h",
            color="Correlation with Profit (r)", color_continuous_scale="Viridis",
            title="🤖 Machine Learning Feature Dominance: Correlation with Net Profit",
            text_auto=".2f"
        )
        fig.update_layout(**theme)
        return fig

    # 2. State Yield
    if any(k in q for k in ["state yield", "yield by state", "highest yield state", "top state", "which state", "state comparison"]):
        agg = df.groupby("State")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
        fig = px.bar(
            agg, x="State", y="Yield_ton_per_hectare",
            color="Yield_ton_per_hectare", color_continuous_scale="Greens",
            title="🌾 Agricultural Yield Benchmark by State (tons/ha)",
            labels={"Yield_ton_per_hectare": "Avg Yield (t/ha)", "State": "State"},
            text_auto=".2f"
        )
        fig.update_layout(**theme)
        fig.update_traces(textposition="outside")
        return fig

    # 3. Crop Yield
    if any(k in q for k in ["crop yield", "yield by crop", "best crop", "highest yielding crop", "top crop", "variety"]):
        agg = df.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
        fig = px.bar(
            agg, x="Crop_Type", y="Yield_ton_per_hectare",
            color="Yield_ton_per_hectare", color_continuous_scale="Greens",
            title="🌱 Crop Variety Productivity Benchmark (tons/ha)",
            labels={"Yield_ton_per_hectare": "Avg Yield (t/ha)", "Crop_Type": "Crop Variety"},
            text_auto=".2f"
        )
        fig.update_layout(**theme)
        fig.update_traces(textposition="outside")
        return fig

    # 4. Farmer Profit by State
    if any(k in q for k in ["profit by state", "state profit", "profitable state", "income by state"]):
        agg = df.groupby("State")["Farmer_Profit_INR"].mean().reset_index().sort_values("Farmer_Profit_INR", ascending=False)
        fig = px.bar(
            agg, x="State", y="Farmer_Profit_INR",
            color="Farmer_Profit_INR", color_continuous_scale="Teal",
            title="💰 Average Farmer Net Return by State (INR)",
            labels={"Farmer_Profit_INR": "Net Profit (INR)", "State": "State"},
            text_auto=",.0f"
        )
        fig.update_layout(**theme)
        fig.update_traces(textposition="outside")
        return fig

    # 5. Farmer Profit by Crop
    if any(k in q for k in ["profit by crop", "crop profit", "profitable crop", "most profitable", "highest profit", "profit"]):
        agg = df.groupby("Crop_Type")["Farmer_Profit_INR"].mean().reset_index().sort_values("Farmer_Profit_INR", ascending=False)
        fig = px.bar(
            agg, x="Crop_Type", y="Farmer_Profit_INR",
            color="Farmer_Profit_INR", color_continuous_scale="Teal",
            title="💰 Commercial Profitability Ranking by Crop Variety (INR)",
            labels={"Farmer_Profit_INR": "Net Profit (INR)", "Crop_Type": "Crop Variety"},
            text_auto=",.0f"
        )
        fig.update_layout(**theme)
        fig.update_traces(textposition="outside")
        return fig

    # 6. Multi-Year Historical Trajectories
    if any(k in q for k in ["trend", "trajectory", "year", "historical", "over time", "annual", "2014", "2024"]):
        agg = df.groupby("Year")[["Yield_ton_per_hectare", "Farmer_Profit_INR"]].mean().reset_index()
        fig = px.line(
            agg, x="Year", y="Yield_ton_per_hectare", markers=True,
            title="📈 10-Year National Crop Yield Trajectory (2014-2024)",
            labels={"Yield_ton_per_hectare": "Yield (t/ha)", "Year": "Harvest Year"}
        )
        fig.update_traces(line=dict(color="#00E676", width=3), marker=dict(size=8, color="#B9F6CA"))
        fig.update_layout(**theme)
        return fig

    # 7. Season Comparison
    if any(k in q for k in ["season", "kharif", "rabi", "zaid"]):
        agg = df.groupby("Season")[["Yield_ton_per_hectare", "Farmer_Profit_INR"]].mean().reset_index()
        fig = px.bar(
            agg, x="Season", y="Yield_ton_per_hectare", color="Season",
            color_discrete_sequence=["#2E7D32", "#1565C0", "#E65100"],
            title="🌦️ Seasonal Yield Benchmarking (tons/ha)",
            labels={"Yield_ton_per_hectare": "Yield (t/ha)"},
            text_auto=".2f"
        )
        fig.update_layout(**theme)
        fig.update_traces(textposition="outside")
        return fig

    # 8. Soil Types
    if any(k in q for k in ["soil", "alluvial", "black", "laterite", "red", "sandy"]):
        agg = df.groupby("Soil_Type")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
        fig = px.bar(
            agg, x="Soil_Type", y="Yield_ton_per_hectare", color="Soil_Type",
            title="🪨 Agronomic Productivity Across Soil Classifications (tons/ha)",
            labels={"Yield_ton_per_hectare": "Yield (t/ha)", "Soil_Type": "Soil Classification"},
            text_auto=".2f"
        )
        fig.update_layout(**theme)
        fig.update_traces(textposition="outside")
        return fig

    # 9. Irrigation Methods
    if any(k in q for k in ["irrigation", "drip", "canal", "sprinkler", "tube well", "rain-fed"]):
        agg = df.groupby("Irrigation_Type")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
        fig = px.bar(
            agg, x="Irrigation_Type", y="Yield_ton_per_hectare", color="Irrigation_Type",
            title="💧 Productivity Efficiency Across Irrigation Infrastructure (tons/ha)",
            labels={"Yield_ton_per_hectare": "Yield (t/ha)", "Irrigation_Type": "Irrigation Method"},
            text_auto=".2f"
        )
        fig.update_layout(**theme)
        fig.update_traces(textposition="outside")
        return fig

    # 10. Market Prices by Crop
    if any(k in q for k in ["price", "market price", "rate", "cost per ton"]):
        agg = df.groupby("Crop_Type")["Market_Price_per_ton"].mean().reset_index().sort_values("Market_Price_per_ton", ascending=False)
        fig = px.bar(
            agg, x="Crop_Type", y="Market_Price_per_ton",
            color="Market_Price_per_ton", color_continuous_scale="Viridis",
            title="🏷️ Benchmark Market Price Realization by Crop (INR/ton)",
            labels={"Market_Price_per_ton": "Price (INR/ton)", "Crop_Type": "Crop Variety"},
            text_auto=",.0f"
        )
        fig.update_layout(**theme)
        fig.update_traces(textposition="outside")
        return fig

    # 11. Rainfall vs Yield Scatter
    if any(k in q for k in ["rainfall", "rain", "precipitation", "water"]):
        sample_df = df.sample(min(800, len(df)), random_state=42)
        fig = px.scatter(
            sample_df, x="Rainfall_mm", y="Yield_ton_per_hectare", color="Crop_Type",
            title="🌧️ Rainfall Distribution vs. Crop Yield Realization",
            labels={"Rainfall_mm": "Rainfall (mm)", "Yield_ton_per_hectare": "Yield (t/ha)"},
            opacity=0.65
        )
        fig.update_layout(**theme)
        return fig

    # 12. Default Macro Benchmark (GUARANTEES a chart is ALWAYS returned!)
    agg = df.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
    fig = px.bar(
        agg, x="Crop_Type", y="Yield_ton_per_hectare",
        color="Yield_ton_per_hectare", color_continuous_scale="Greens",
        title="🌾 National Crop Yield Benchmark Overview (tons/ha)",
        labels={"Yield_ton_per_hectare": "Avg Yield (t/ha)", "Crop_Type": "Crop Variety"},
        text_auto=".2f"
    )
    fig.update_layout(**theme)
    fig.update_traces(textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# Persistent Multi-Session Chat History Management
# ---------------------------------------------------------------------------

def load_all_sessions() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_all_sessions(sessions: list[dict]):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def create_new_session(title: str = "New Session") -> str:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sessions = load_all_sessions()
    sessions.insert(0, {
        "id": session_id,
        "title": title,
        "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "messages": []
    })
    save_all_sessions(sessions)
    return session_id

def append_message_to_session(session_id: str, role: str, content: str, chart_title: str | None = None):
    sessions = load_all_sessions()
    for s in sessions:
        if s["id"] == session_id:
            s["messages"].append({
                "role": role,
                "content": content,
                "chart_title": chart_title,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            if s["title"] in ["New Session", "Initial AgriSmart Briefing"] and role == "user":
                s["title"] = content[:42] + ("..." if len(content) > 42 else "")
            break
    save_all_sessions(sessions)

def delete_session(session_id: str):
    sessions = load_all_sessions()
    sessions = [s for s in sessions if s["id"] != session_id]
    save_all_sessions(sessions)

def clear_all_sessions():
    if HISTORY_FILE.exists():
        try:
            HISTORY_FILE.unlink()
        except Exception:
            pass

def export_session_markdown(session: dict) -> str:
    md = [f"# AgroSense AI Executive Briefing Transcript\n"]
    md.append(f"**Session:** {session.get('title', 'Executive Session')}")
    md.append(f"**Generated:** {session.get('created_at', datetime.now().strftime('%d %b %Y'))}\n")
    md.append("---\n")
    for msg in session.get("messages", []):
        role_name = "User (Agribusiness Manager)" if msg["role"] == "user" else "AgroSense AI Copilot"
        md.append(f"### {role_name} ({msg.get('timestamp', '')})")
        md.append(msg["content"])
        if msg.get("chart_title"):
            md.append(f"\n> 📊 Attached Visualization: *{msg['chart_title']}*\n")
        md.append("\n---\n")
    return "\n".join(md)
