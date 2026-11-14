"""
AgroSense AI — Advanced Enterprise RAG Backend
===============================================
Features:
  - Multi-tiered Knowledge Chunks (115+ domain vectors)
  - FAISS Vector Database with normalized TF-IDF embeddings (sub-ms retrieval)
  - Groq Cloud LLM (openai/gpt-oss-20b / 120b) with zero-hallucination constraint
  - Context-Aware Dynamic Plotly Chart Engine (Bar, Line, Scatter, Heatmap)
  - Multi-session Chat History Manager (Read, Resume, Delete individual/all, Export)
"""

import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment
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

SYSTEM_PROMPT = """You are AgroSense AI, the Senior Agricultural Intelligence & Decision-Support Copilot for AgriSmart Dashboard.

OBJECTIVE:
Provide elite, management-level briefings to agricultural directors, agribusiness strategists, and farm operators.

MANDATORY RULES:
1. STRICT ZERO-HALLUCINATION: Base every number, state, crop, and trend ONLY on the DATA CONTEXT provided below. Do not guess or extrapolate unrecorded figures.
2. If data for a specific detail is not present, state with clarity: "Based on the AgriSmart historical records (2014-2024), specific data for this exact query is not cataloged. The closest recorded metrics are..."
3. EXECUTIVE MANAGEMENT STRUCTURE: Always format answers into structured sections:
   - 📊 **Executive Summary**: Core finding, key metric, and direct answer in 2-3 bullet points.
   - 🔍 **Quantitative Benchmark & Analysis**: Detailed comparative breakdown using clear Markdown tables or bulleted KPIs.
   - 📋 **Strategic Action Plan**: 2-4 concrete, management-level recommendations (resource allocation, agronomic advisory, risk mitigation).
4. Tone must be professional, authoritative, and data-backed.
"""

def _build_comprehensive_chunks(df: pd.DataFrame) -> list[str]:
    chunks = []

    # 1. Macro Overview
    chunks.append(
        f"[Macro Benchmark] AgriSmart Historical Repository (2014-2024): Total 7,920 records. "
        f"8 States: {', '.join(sorted(df['State'].unique()))}. "
        f"8 Crops: {', '.join(sorted(df['Crop_Type'].unique()))}. "
        f"3 Seasons: Kharif, Rabi, Zaid. "
        f"National Mean Yield: {df['Yield_ton_per_hectare'].mean():.2f} tons/ha (Min: {df['Yield_ton_per_hectare'].min():.2f}, Max: {df['Yield_ton_per_hectare'].max():.2f}). "
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

    # 8. State-Crop Combinations (Top 64 combos)
    for (state, crop), grp in df.groupby(["State", "Crop_Type"]):
        chunks.append(
            f"[State-Crop Combination] State: {state}, Crop: {crop}. Average Yield: {grp['Yield_ton_per_hectare'].mean():.2f} t/ha. "
            f"Average Profit: INR {grp['Farmer_Profit_INR'].mean():,.0f}."
        )

    # 9. Key Econometric & Agronomic Discoveries
    chunks.append(
        "[Agronomic Discovery: Yield Drivers] In this dataset, cultivated area (r=0.69) and crop yield (r=0.55) "
        "are the primary determinants of farmer profit. Yield variance across weather/soil factors is uniformly distributed (~4.8 t/ha), "
        "making post-harvest price realization and operational scale the key strategic levers for commercial solvency."
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

def chat(user_query: str, chat_history: list[dict]) -> tuple[str, go.Figure | None]:
    vs = get_vector_store()
    retrieved_chunks = vs.retrieve(user_query, top_k=TOP_K)
    context_str = "\n".join(f"• {c}" for c in retrieved_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent context window
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    augmented_prompt = (
        f"=== GROUND TRUTH AGRISMART DATA REPOSITORY (2014-2024) ===\n"
        f"{context_str}\n\n"
        f"=== USER QUERY ===\n"
        f"{user_query}\n\n"
        f"Deliver a management-level executive briefing strictly based on the ground truth above."
    )
    messages.append({"role": "user", "content": augmented_prompt})

    answer = call_groq(messages)
    fig = generate_dynamic_chart(user_query)

    return answer, fig

def generate_dynamic_chart(query: str) -> go.Figure | None:
    """Generates an executive-grade interactive Plotly chart styled in Emerald/Forest Dark theme."""
    q = query.lower()
    df = get_df()

    theme = dict(
        template="plotly_dark",
        paper_bgcolor="#0F1713",
        plot_bgcolor="#0F1713",
        font=dict(family="Arial, sans-serif", color="#E0E6E2"),
        margin=dict(l=40, r=40, t=60, b=50),
    )

    # 1. State Yield
    if any(k in q for k in ["yield by state", "state yield", "highest yield state", "top state", "state performance", "which state"]):
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

    # 2. Crop Yield
    if any(k in q for k in ["yield by crop", "crop yield", "best crop", "highest yielding crop", "top crop"]):
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

    # 3. Farmer Profit by State
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

    # 4. Farmer Profit by Crop
    if any(k in q for k in ["profit by crop", "crop profit", "profitable crop", "most profitable", "highest profit"]):
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

    # 5. Multi-Year Historical Trajectories
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

    # 6. Season Comparison
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

    # 7. Soil Types
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

    # 8. Irrigation Methods
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

    # 9. Market Prices by Crop
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

    # 10. Rainfall vs Yield Scatter
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

    # 11. State x Crop Matrix Heatmap
    if any(k in q for k in ["heatmap", "matrix", "cross", "comparison table"]):
        pivot = df.pivot_table(values="Yield_ton_per_hectare", index="State", columns="Crop_Type", aggfunc="mean")
        fig = px.imshow(
            pivot, color_continuous_scale="Greens",
            title="🗺️ Productivity Heatmap: State × Crop Yield Matrix (tons/ha)",
            labels={"color": "Yield (t/ha)"}
        )
        fig.update_layout(**theme)
        return fig

    return None


# ---------------------------------------------------------------------------
# Persistent Multi-Session Chat History Management
# ---------------------------------------------------------------------------

def load_all_sessions() -> list[dict]:
    """Load persistent chat history from disk."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_all_sessions(sessions: list[dict]):
    """Persist chat sessions to disk."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def create_new_session(title: str = "New Session") -> str:
    """Creates a new session and returns its ID."""
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
    """Appends a message to the specified session."""
    sessions = load_all_sessions()
    for s in sessions:
        if s["id"] == session_id:
            s["messages"].append({
                "role": role,
                "content": content,
                "chart_title": chart_title,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            if s["title"] == "New Session" and role == "user":
                s["title"] = content[:45] + ("..." if len(content) > 45 else "")
            break
    save_all_sessions(sessions)

def delete_session(session_id: str):
    """Deletes a specific session."""
    sessions = load_all_sessions()
    sessions = [s for s in sessions if s["id"] != session_id]
    save_all_sessions(sessions)

def clear_all_sessions():
    """Deletes all chat history."""
    if HISTORY_FILE.exists():
        try:
            HISTORY_FILE.unlink()
        except Exception:
            pass

def export_session_markdown(session: dict) -> str:
    """Converts a session into a formatted Markdown report."""
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
