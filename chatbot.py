"""
AgroSense AI — Agricultural Intelligence RAG Backend
=====================================================
Architecture:
  1. CSV Dataset -> Structured Knowledge Chunks
  2. TF-IDF Dense Normalized Embeddings -> FAISS Vector Database (IndexFlatIP)
  3. Top-k Semantic Similarity Retrieval
  4. Groq LLM (openai/gpt-oss-20b / openai/gpt-oss-120b)
  5. Executive Advisory System Prompt (Zero Hallucination)
  6. Context-Aware Management Plotly Visualizations

The Groq API key is loaded strictly from .env and NEVER exposed in the UI.
"""

import os
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
TOP_K = 6

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

SYSTEM_PROMPT = """You are AgroSense AI, the Senior Executive Agricultural Intelligence Advisor for the AgriSmart platform.

YOUR OBJECTIVE:
Provide strategic, management-level decision support to agricultural policymakers, agribusiness leaders, and farm managers.

CRITICAL RULES:
1. STRICT ZERO-HALLUCINATION: Base your answers ONLY on the provided DATA CONTEXT below. Do NOT invent figures, states, or crops not in the data.
2. If the user asks something not present in the dataset, state clearly: "Based on the AgriSmart historical dataset (2014-2024), specific data for this query is not available; however, the following relevant metrics are recorded..."
3. TONE & STYLE: Use executive management language. Organize your response with clear sections:
   - 📊 **Executive Summary** (Key metrics & direct answer)
   - 🔍 **In-Depth Quantitative Analysis** (State/crop comparisons, financial or agronomic breakdown)
   - 📋 **Strategic Management Recommendations** (Actionable steps for yield optimization, risk mitigation, or profit maximization)
4. Use formatted tables, bullet points, and bold text for readability.
"""

def _build_knowledge_chunks(df: pd.DataFrame) -> list[str]:
    chunks = []

    # 1. State-level summaries
    for state, grp in df.groupby("State"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        avg_r = grp["Rainfall_mm"].mean()
        top_crop = grp.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().idxmax()
        top_crop_val = grp.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().max()
        most_prof_crop = grp.groupby("Crop_Type")["Farmer_Profit_INR"].mean().idxmax()
        most_prof_val = grp.groupby("Crop_Type")["Farmer_Profit_INR"].mean().max()
        chunks.append(
            f"[State Profile] State: {state}. Average Yield: {avg_y:.2f} tons/ha. "
            f"Average Farmer Profit: INR {avg_p:,.0f}. Mean Rainfall: {avg_r:.1f} mm. "
            f"Highest yielding crop in {state} is {top_crop} ({top_crop_val:.2f} t/ha). "
            f"Most profitable crop in {state} is {most_prof_crop} (INR {most_prof_val:,.0f})."
        )

    # 2. Crop-level summaries
    for crop, grp in df.groupby("Crop_Type"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        avg_fert = grp["Fertilizer_Used_kg"].mean()
        avg_pest = grp["Pesticide_Used_kg"].mean()
        avg_price = grp["Market_Price_per_ton"].mean()
        best_state = grp.groupby("State")["Yield_ton_per_hectare"].mean().idxmax()
        chunks.append(
            f"[Crop Profile] Crop: {crop}. National Average Yield: {avg_y:.2f} tons/ha. "
            f"Average Profit: INR {avg_p:,.0f}. Average Market Price: INR {avg_price:,.0f}/ton. "
            f"Average Fertilizer: {avg_fert:.1f} kg. Average Pesticide: {avg_pest:.1f} kg. "
            f"Top performing state for {crop} by yield is {best_state}."
        )

    # 3. Season-level summaries
    for season, grp in df.groupby("Season"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        top_c = grp.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().idxmax()
        chunks.append(
            f"[Season Profile] Season: {season}. Average Yield: {avg_y:.2f} tons/ha. "
            f"Average Farmer Profit: INR {avg_p:,.0f}. Top yielding crop in {season} is {top_c}."
        )

    # 4. Soil-type summaries
    for soil, grp in df.groupby("Soil_Type"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        chunks.append(
            f"[Soil Analysis] Soil Type: {soil}. Average Yield: {avg_y:.2f} tons/ha. "
            f"Average Farmer Profit: INR {avg_p:,.0f}."
        )

    # 5. Irrigation-type summaries
    for irr, grp in df.groupby("Irrigation_Type"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        chunks.append(
            f"[Irrigation Analysis] Irrigation Method: {irr}. Average Yield: {avg_y:.2f} tons/ha. "
            f"Average Farmer Profit: INR {avg_p:,.0f}."
        )

    # 6. Annual Trends (2014-2024)
    for year, grp in df.groupby("Year"):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        chunks.append(
            f"[Annual Trend] Year {year}: Average Yield: {avg_y:.2f} tons/ha. "
            f"Average Farmer Profit: INR {avg_p:,.0f}."
        )

    # 7. State-Crop Cross Matrix
    for (state, crop), grp in df.groupby(["State", "Crop_Type"]):
        avg_y = grp["Yield_ton_per_hectare"].mean()
        avg_p = grp["Farmer_Profit_INR"].mean()
        chunks.append(
            f"[State-Crop Matrix] In {state}, crop {crop} yields {avg_y:.2f} tons/ha "
            f"generating average net profit of INR {avg_p:,.0f}."
        )

    # 8. Overall Macro Summary
    chunks.append(
        f"[Macro Dataset Overview] Dataset encompasses {len(df)} records across 8 states "
        f"({', '.join(sorted(df['State'].unique()))}) and 8 crops "
        f"({', '.join(sorted(df['Crop_Type'].unique()))}) covering 2014 to 2024. "
        f"National Mean Yield: {df['Yield_ton_per_hectare'].mean():.2f} tons/ha. "
        f"National Mean Profit: INR {df['Farmer_Profit_INR'].mean():,.0f}. "
        f"Highest Yielding State: {df.groupby('State')['Yield_ton_per_hectare'].mean().idxmax()} "
        f"({df.groupby('State')['Yield_ton_per_hectare'].mean().max():.2f} tons/ha). "
        f"Most Profitable Crop: {df.groupby('Crop_Type')['Farmer_Profit_INR'].mean().idxmax()} "
        f"(INR {df.groupby('Crop_Type')['Farmer_Profit_INR'].mean().max():,.0f})."
    )

    return chunks


class AgroFAISSStore:
    """FAISS vector database with TF-IDF vector embeddings for sub-millisecond retrieval."""

    def __init__(self):
        self.chunks: list[str] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=3000)
        self.index = None

    def build(self, chunks: list[str]):
        self.chunks = chunks
        tfidf_mat = self.vectorizer.fit_transform(chunks).astype(np.float32)
        dense_embs = tfidf_mat.toarray()
        # Normalize vectors for cosine similarity via inner product
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


_store: AgroFAISSStore | None = None
_df_cache: pd.DataFrame | None = None

def get_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(CSV_PATH)
    return _df_cache

def get_vector_store() -> AgroFAISSStore:
    global _store
    if _store is None or _store.index is None:
        df = get_df()
        chunks = _build_knowledge_chunks(df)
        vs = AgroFAISSStore()
        vs.build(chunks)
        _store = vs
    return _store

def check_availability() -> dict:
    return {
        "faiss": _FAISS_AVAILABLE,
        "groq": _GROQ_AVAILABLE,
        "api_key_set": bool(_GROQ_API_KEY),
    }

def call_groq(messages: list[dict]) -> str:
    if not _GROQ_AVAILABLE:
        raise ImportError("groq is not installed. Run: pip install groq")
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
                max_tokens=1500,
                top_p=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All Groq model attempts failed. Last error: {last_err}")

def chat(user_query: str, chat_history: list[dict]) -> tuple[str, go.Figure | None]:
    vs = get_vector_store()
    retrieved_chunks = vs.retrieve(user_query, top_k=TOP_K)
    context_str = "\n".join(f"• {c}" for c in retrieved_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    augmented_prompt = (
        f"=== GROUND TRUTH DATA CONTEXT (AGRISMART 2014-2024) ===\n"
        f"{context_str}\n\n"
        f"=== USER QUERY ===\n"
        f"{user_query}\n\n"
        f"Provide an executive briefing strictly based on the ground truth above."
    )
    messages.append({"role": "user", "content": augmented_prompt})

    answer = call_groq(messages)
    fig = _generate_chart(user_query)

    return answer, fig

def _generate_chart(query: str) -> go.Figure | None:
    q = query.lower()
    df = get_df()

    # State Yield Comparison
    if any(k in q for k in ["yield by state", "state yield", "highest yield state", "top state", "state performance", "which state"]):
        agg = df.groupby("State")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
        fig = px.bar(
            agg, x="State", y="Yield_ton_per_hectare",
            color="Yield_ton_per_hectare", color_continuous_scale="Greens",
            title="🌾 Average Agricultural Yield by State (tons/hectare)",
            labels={"Yield_ton_per_hectare": "Yield (tons/ha)", "State": "State"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111B15", plot_bgcolor="#111B15", font_color="#E0E6E2")
        return fig

    # Crop Yield Comparison
    if any(k in q for k in ["yield by crop", "crop yield", "best crop", "highest yielding crop", "top crop"]):
        agg = df.groupby("Crop_Type")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
        fig = px.bar(
            agg, x="Crop_Type", y="Yield_ton_per_hectare",
            color="Yield_ton_per_hectare", color_continuous_scale="Greens",
            title="🌱 Average Crop Yield by Variety (tons/hectare)",
            labels={"Yield_ton_per_hectare": "Yield (tons/ha)", "Crop_Type": "Crop Variety"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111B15", plot_bgcolor="#111B15", font_color="#E0E6E2")
        return fig

    # Farmer Profit by State
    if any(k in q for k in ["profit by state", "state profit", "profitable state", "income by state"]):
        agg = df.groupby("State")["Farmer_Profit_INR"].mean().reset_index().sort_values("Farmer_Profit_INR", ascending=False)
        fig = px.bar(
            agg, x="State", y="Farmer_Profit_INR",
            color="Farmer_Profit_INR", color_continuous_scale="Teal",
            title="💰 Average Farmer Net Profit by State (INR)",
            labels={"Farmer_Profit_INR": "Net Profit (INR)", "State": "State"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111B15", plot_bgcolor="#111B15", font_color="#E0E6E2")
        return fig

    # Farmer Profit by Crop
    if any(k in q for k in ["profit by crop", "profitable crop", "most profitable", "crop profit", "highest profit"]):
        agg = df.groupby("Crop_Type")["Farmer_Profit_INR"].mean().reset_index().sort_values("Farmer_Profit_INR", ascending=False)
        fig = px.bar(
            agg, x="Crop_Type", y="Farmer_Profit_INR",
            color="Farmer_Profit_INR", color_continuous_scale="Teal",
            title="💰 Average Farmer Net Profit by Crop (INR)",
            labels={"Farmer_Profit_INR": "Net Profit (INR)", "Crop_Type": "Crop Variety"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111B15", plot_bgcolor="#111B15", font_color="#E0E6E2")
        return fig

    # Yearly Trends
    if any(k in q for k in ["trend", "year", "historical", "over time", "annual"]):
        agg = df.groupby("Year")[["Yield_ton_per_hectare", "Farmer_Profit_INR"]].mean().reset_index()
        fig = px.line(
            agg, x="Year", y="Yield_ton_per_hectare", markers=True,
            title="📈 Historical National Crop Yield Trajectory (2014-2024)",
            labels={"Yield_ton_per_hectare": "Yield (tons/ha)", "Year": "Year"}
        )
        fig.update_traces(line_color="#00C853", marker=dict(size=8, color="#B9F6CA"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111B15", plot_bgcolor="#111B15", font_color="#E0E6E2")
        return fig

    # Season Breakdown
    if any(k in q for k in ["season", "kharif", "rabi", "zaid"]):
        agg = df.groupby("Season")["Yield_ton_per_hectare"].mean().reset_index()
        fig = px.bar(
            agg, x="Season", y="Yield_ton_per_hectare", color="Season",
            color_discrete_sequence=["#2E7D32", "#1565C0", "#E65100"],
            title="🌦️ Seasonal Productivity Comparison (tons/hectare)",
            labels={"Yield_ton_per_hectare": "Yield (tons/ha)"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111B15", plot_bgcolor="#111B15", font_color="#E0E6E2")
        return fig

    # Soil Type
    if any(k in q for k in ["soil", "alluvial", "black", "laterite", "red", "sandy"]):
        agg = df.groupby("Soil_Type")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
        fig = px.bar(
            agg, x="Soil_Type", y="Yield_ton_per_hectare", color="Soil_Type",
            title="🪨 Agronomic Productivity Across Soil Classifications",
            labels={"Yield_ton_per_hectare": "Yield (tons/ha)", "Soil_Type": "Soil Type"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111B15", plot_bgcolor="#111B15", font_color="#E0E6E2")
        return fig

    # Irrigation Type
    if any(k in q for k in ["irrigation", "drip", "canal", "sprinkler", "tube well", "rain-fed"]):
        agg = df.groupby("Irrigation_Type")["Yield_ton_per_hectare"].mean().reset_index().sort_values("Yield_ton_per_hectare", ascending=False)
        fig = px.bar(
            agg, x="Irrigation_Type", y="Yield_ton_per_hectare", color="Irrigation_Type",
            title="💧 Yield Efficiency Across Irrigation Infrastructure",
            labels={"Yield_ton_per_hectare": "Yield (tons/ha)", "Irrigation_Type": "Irrigation Method"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111B15", plot_bgcolor="#111B15", font_color="#E0E6E2")
        return fig

    return None
