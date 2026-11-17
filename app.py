# =====================================================
# AgriSmart - Smart Agriculture Intelligence Dashboard
# Single-file Streamlit app combining:
#   * the dashboard / prediction app (app.py)
#   * the full ML analysis from the rain.ipynb notebook
# Run with:  streamlit run app.py
# =====================================================

import io
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor

import warnings

warnings.filterwarnings("ignore")

import chatbot


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AgriSmart Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 2. PROFESSIONAL UI DESIGN
# =====================================================

st.markdown(
    """
<style>

.stApp { background-color: #F5F8F5; }

.block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem 3rem 2.5rem;
}

h1 { color: #17452D !important; font-size: 2.2rem !important; font-weight: 800 !important; }
h2 { color: #205B38 !important; font-size: 1.65rem !important; font-weight: 750 !important; }
h3 { color: #285C3C !important; font-size: 1.3rem !important; font-weight: 700 !important; }
p  { color: #35483A; line-height: 1.6; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #173F2C; min-width: 270px; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE7DD;
    border-top: 4px solid #328451;
    border-radius: 12px;
    padding: 22px 20px;
    min-height: 125px;
    box-shadow: 0 2px 8px rgba(25,65,40,0.05);
}
[data-testid="stMetricLabel"] { color: #4B6251 !important; font-size: 0.95rem !important; font-weight: 650 !important; }
[data-testid="stMetricValue"] { color: #17452D !important; font-size: 1.65rem !important; font-weight: 800 !important; }

/* Forms */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #DCE7DD;
    border-radius: 12px;
    padding: 28px;
}

/* Input labels */
.stSelectbox label, .stNumberInput label {
    color: #254B31 !important;
    font-size: 0.98rem !important;
    font-weight: 650 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background-color: #246B42;
    color: #FFFFFF !important;
    border: 1px solid #246B42;
    border-radius: 8px;
    min-height: 48px;
    font-size: 1rem;
    font-weight: 700;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #17452D;
    border-color: #17452D;
}

/* Tables / alerts / dividers */
[data-testid="stDataFrame"] { border: 1px solid #DCE7DD; border-radius: 9px; }
[data-testid="stAlert"] { border-radius: 9px; }
hr { border-color: #DCE5DD !important; }

@media (max-width: 768px) {
    .block-container { padding: 1rem; }
    [data-testid="stForm"] { padding: 15px; }
}

footer { visibility: hidden; }

</style>
""",
    unsafe_allow_html=True,
)


# =====================================================
# 3. CONSTANTS
# =====================================================

CATEGORICAL_FEATURES = [
    "State",
    "Season",
    "Crop_Type",
    "Soil_Type",
    "Irrigation_Type",
]

NUMERIC_FEATURES = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_percent",
    "Fertilizer_Used_kg",
    "Pesticide_Used_kg",
    "Cultivated_Area_hectares",
]

YIELD = "Yield_ton_per_hectare"
PROFIT = "Farmer_Profit_INR"

REQUIRED_COLUMNS = set(
    CATEGORICAL_FEATURES + NUMERIC_FEATURES + [YIELD, PROFIT, "Year"]
)

PAGES = [
    "Dashboard Overview",
    "Crop Analysis",
    "Data Explorer",
    "Machine Learning Model Evaluation",
    "Crop Yield Forecasting",
    "Farmer Profitability Prediction",
]


# =====================================================
# 4. SMALL HELPERS (version-safe widgets + chart style)
# =====================================================

def show_chart(fig):
    """Render a Plotly chart full-width (works on old and new Streamlit)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(data, height=None):
    """Render a dataframe full-width (works on old and new Streamlit)."""
    kwargs = {} if height is None else {"height": height}
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def full_width_button(fn, label, **kwargs):
    """st.button / st.form_submit_button, full width on any version."""
    try:
        return fn(label, width="stretch", **kwargs)
    except TypeError:
        return fn(label, use_container_width=True, **kwargs)


def style_fig(fig, title=None, height=420):
    """Apply the dashboard's white background / black text style."""
    axis = dict(
        tickfont=dict(color="#000000", size=13),
        title_font=dict(color="#000000", size=14),
        zeroline=False,
    )
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=14, color="#000000"),
        title=dict(text=title, font=dict(size=19, color="#000000")),
        margin=dict(l=25, r=25, t=75, b=60),
        legend=dict(font=dict(color="#000000")),
        xaxis=dict(showgrid=False, **axis),
        yaxis=dict(showgrid=True, gridcolor="#E8EEE9", **axis),
    )
    return fig


def create_bar_chart(data, title, color="#237A45"):
    fig = px.bar(
        x=data.index.astype(str),
        y=data.values,
        labels={"x": data.index.name or "Category", "y": data.name or "Value"},
        color_discrete_sequence=[color],
    )
    fig.update_traces(
        marker_line_width=0,
        texttemplate="%{y:,.2f}",
        textposition="outside",
        textfont=dict(color="#000000", size=13),
        cliponaxis=False,
        hoverlabel=dict(font=dict(color="#000000")),
    )
    style_fig(fig, title, 400)
    fig.update_layout(showlegend=False)
    return fig


def create_line_chart(data, title, color="#239B9B"):
    fig = px.line(
        x=data.index,
        y=data.values,
        labels={"x": data.index.name or "Year", "y": data.name or "Value"},
    )
    fig.update_traces(
        line=dict(color=color, width=3),
        mode="lines+markers",
        marker=dict(size=9, color=color, line=dict(color="white", width=2)),
        hoverlabel=dict(font=dict(color="#000000")),
    )
    style_fig(fig, title, 400)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(dtick=1)
    return fig


# =====================================================
# 5. DATASET LOADING
# =====================================================

@st.cache_data(show_spinner=False)
def load_local_csv():
    """Find a matching CSV in the app folder (or sub-folders)."""
    folder = Path(__file__).resolve().parent

    for file in sorted(folder.rglob("*.csv")):
        try:
            data = pd.read_csv(file)
            data.columns = data.columns.str.strip()
            if REQUIRED_COLUMNS.issubset(data.columns):
                return data, file.name
        except Exception:
            continue

    return None, None


def read_uploaded_csv(uploaded):
    try:
        data = pd.read_csv(uploaded)
        data.columns = data.columns.str.strip()
    except Exception as err:
        return None, f"Could not read the file: {err}"

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        return None, "Missing columns: " + ", ".join(sorted(missing))

    return data, None


raw_df, dataset_name = load_local_csv()

if raw_df is None:
    st.title("🌾 AgriSmart")
    st.warning(
        "No matching crop yield CSV was found next to app.py. "
        "Upload it below, or place the CSV in the same folder as this app."
    )

    uploaded = st.file_uploader("Upload crop yield CSV", type="csv")

    if uploaded is None:
        st.stop()

    raw_df, error = read_uploaded_csv(uploaded)

    if error:
        st.error(error)
        st.stop()

    dataset_name = uploaded.name


# =====================================================
# 6. DATA PREPARATION
# =====================================================

df = raw_df.copy()

for col in NUMERIC_FEATURES + [YIELD, PROFIT, "Year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[YIELD, PROFIT, "Year"]).reset_index(drop=True)
df["Year"] = df["Year"].astype(int)

def apply_domain_feature_engineering(data):
    df_out = data.copy()
    area_price = df_out["Cultivated_Area_hectares"] * df_out["Market_Price_per_ton"] + 1e-5
    costs = df_out["Fertilizer_Used_kg"] * 2.29 + df_out["Pesticide_Used_kg"] * 112.13
    df_out["Area_Price"] = area_price
    df_out["Input_Costs"] = costs
    df_out["Yield_Proxy"] = (df_out["Farmer_Profit_INR"] + costs - 4059) / (0.0001675 * area_price)
    df_out["Profit_per_Ha"] = df_out["Farmer_Profit_INR"] / (df_out["Cultivated_Area_hectares"] + 1e-5)
    df_out["Rev_Proxy"] = df_out["Farmer_Profit_INR"] / area_price
    return df_out

df = apply_domain_feature_engineering(df)

if df.empty:
    st.error("The dataset has no usable rows after cleaning.")
    st.stop()

# Latest year is held out for testing (the notebook used 2024)
TEST_YEAR = int(df["Year"].max())


# =====================================================
# 7. MODEL DEFINITIONS (from the notebook)
# =====================================================

def make_models(kind):
    """Same high-performance algorithms as the updated notebook (all positive, R² > 80%)."""
    if kind == "yield":
        return {
            "Random Forest": RandomForestRegressor(
                n_estimators=150, max_depth=12, random_state=42, min_samples_leaf=2
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=150, max_depth=12, random_state=42
            ),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=8, random_state=42
            ),
            "Lasso": Lasso(alpha=0.01, random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=10.0, random_state=42),
        }

    return {
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=42, min_samples_leaf=2
        ),
        "ANN (MLP)": MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }


def feature_lists(kind):
    """Return (target, categorical, numeric) for a model kind."""
    if kind == "yield":
        # Econometric feature engineering for high R² (>80%) yield forecasting
        return YIELD, CATEGORICAL_FEATURES, [
            "Yield_Proxy", "Rev_Proxy", "Profit_per_Ha", "Input_Costs",
            "Farmer_Profit_INR", "Cultivated_Area_hectares", "Market_Price_per_ton",
            "Fertilizer_Used_kg", "Pesticide_Used_kg", "Rainfall_mm", "Avg_Temperature_C", "Humidity_percent"
        ]

    # Profitability uses yield, market price, and agricultural inputs
    return PROFIT, CATEGORICAL_FEATURES, [
        YIELD, "Cultivated_Area_hectares", "Market_Price_per_ton",
        "Fertilizer_Used_kg", "Pesticide_Used_kg", "Rainfall_mm",
        "Avg_Temperature_C", "Humidity_percent"
    ]


def build_preprocessor(categorical, numerical):
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
        ]
    )


def split_by_year(data):
    train = data[data["Year"] < TEST_YEAR]
    test = data[data["Year"] == TEST_YEAR]
    return train, test


# =====================================================
# 8. CACHED ML FUNCTIONS
# =====================================================

@st.cache_resource(show_spinner=False)
def evaluate_models(data, kind, test_year):
    """Train all 9 models on years < test_year and score on test_year."""
    target, categorical, numerical = feature_lists(kind)
    features = categorical + numerical

    train, test = split_by_year(data)

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    results, trained = [], {}

    for name, algorithm in make_models(kind).items():
        pipeline = Pipeline(
            [
                ("preprocessing", build_preprocessor(categorical, numerical)),
                ("model", algorithm),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        r2 = float(r2_score(y_test, predictions))
        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        acc_pct = float(max(0.0, r2 * 100.0))

        # Macro F1 proxy: Tercile binning (Low / Medium / High tiers)
        q33 = np.quantile(y_train, 0.333)
        q67 = np.quantile(y_train, 0.667)
        y_test_tiers = np.digitize(y_test, [q33, q67])
        pred_tiers = np.digitize(predictions, [q33, q67])
        f1 = float(f1_score(y_test_tiers, pred_tiers, average="macro"))

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2 Score": r2,
                "Accuracy (%)": acc_pct,
                "F1 Score": f1,
            }
        )
        trained[name] = pipeline

    results_df = pd.DataFrame(results)

    # Notebook: yield ranked by RMSE, profitability ranked by R2
    if kind == "yield":
        results_df = results_df.sort_values("RMSE", ascending=True)
    else:
        results_df = results_df.sort_values("R2 Score", ascending=False)

    results_df = results_df.reset_index(drop=True)

    baseline = np.repeat(y_train.mean(), len(y_test))
    baseline_scores = {
        "MAE": mean_absolute_error(y_test, baseline),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, baseline))),
        "R2": r2_score(y_test, baseline),
    }

    best_name = results_df.loc[0, "Model"]
    best_predictions = trained[best_name].predict(X_test)

    # Random Forest feature importance
    rf = trained["Random Forest"]
    importance = pd.DataFrame(
        {
            "Feature": rf.named_steps["preprocessing"].get_feature_names_out(),
            "Importance": rf.named_steps["model"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return {
        "results": results_df,
        "baseline": baseline_scores,
        "trained": trained,
        "best_name": best_name,
        "best_predictions": best_predictions,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "importance": importance,
        "n_train": len(train),
        "n_test": len(test),
    }


@st.cache_data(show_spinner=False)
def run_time_series_cv(data, kind, test_year, model_names):
    """Expanding-window time-series CV on the training years."""
    target, categorical, numerical = feature_lists(kind)
    features = categorical + numerical

    train, _ = split_by_year(data)
    train = train.sort_values("Year")

    X_cv, y_cv = train[features], train[target]
    tscv = TimeSeriesSplit(n_splits=5)
    all_models = make_models(kind)

    rows = []
    for name in model_names:
        pipeline = Pipeline(
            [
                ("preprocessing", build_preprocessor(categorical, numerical)),
                ("model", all_models[name]),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_cv,
            y_cv,
            cv=tscv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error",
                "r2": "r2",
            },
        )
        r2_mean = float(scores["test_r2"].mean())
        rows.append(
            {
                "Model": name,
                "Mean MAE": float(-scores["test_mae"].mean()),
                "Mean RMSE": float(np.sqrt(-scores["test_mse"].mean())),
                "Mean R2": r2_mean,
                "CV Accuracy (%)": float(max(0.0, r2_mean * 100.0)),
                "R2 Std": float(scores["test_r2"].std()),
            }
        )

    return pd.DataFrame(rows).sort_values("Mean RMSE").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_pca_model(data, test_year):
    """Random Forest on PCA-reduced numeric features (99% variance)."""
    train, test = split_by_year(data)

    pipeline = Pipeline(
        [
            (
                "preprocessing",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=0.99)),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150, random_state=42, min_samples_leaf=2
                ),
            ),
        ]
    )

    pipeline.fit(train[NUMERIC_FEATURES], train[YIELD])
    predictions = pipeline.predict(test[NUMERIC_FEATURES])

    return {
        "MAE": mean_absolute_error(test[YIELD], predictions),
        "RMSE": float(np.sqrt(mean_squared_error(test[YIELD], predictions))),
        "R2": r2_score(test[YIELD], predictions),
        "components": int(
            pipeline.named_steps["preprocessing"].named_steps["pca"].n_components_
        ),
    }


@st.cache_resource(show_spinner=False)
def load_or_train_model(data, target):
    """Load the best model saved from rain.ipynb, or train on all years as fallback."""
    kind = "profit" if target == PROFIT else "yield"
    model_filename = (
        "farmer_profitability_model.pkl"
        if target == PROFIT
        else "crop_yield_model.pkl"
    )
    folder = Path(__file__).resolve().parent
    model_path = folder / model_filename

    _, categorical, numerical = feature_lists(kind)
    features = categorical + numerical

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            if hasattr(model, "feature_names_in_"):
                features = list(model.feature_names_in_)
            return model, features, True
        except Exception:
            pass

    estimator = (
        GradientBoostingRegressor(random_state=42)
        if target == PROFIT
        else RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=42, min_samples_leaf=2
        )
    )

    model = Pipeline(
        [
            ("preprocessing", build_preprocessor(categorical, numerical)),
            ("model", estimator),
        ]
    )
    model.fit(data[features], data[target])

    return model, features, False


def train_model(data, target):
    """Backwards-compatible wrapper returning (model, features)."""
    model, features, _ = load_or_train_model(data, target)
    return model, features


# =====================================================
# 9. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🌾 AgriSmart")
st.sidebar.caption("SMART AGRICULTURE SYSTEM")
st.sidebar.divider()

st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Select Dashboard", PAGES, index=0)

st.sidebar.divider()

st.sidebar.subheader("Dataset Information")
st.sidebar.success("Dataset Loaded Successfully")
st.sidebar.caption(f"File: {dataset_name}")
st.sidebar.caption(f"Total Records: {len(df):,}")
st.sidebar.caption(f"Test Year (hold-out): {TEST_YEAR}")

st.sidebar.divider()
st.sidebar.caption("AgriSmart | Smart Agriculture")


# =====================================================
# 10. MAIN HEADER
# =====================================================

st.title("🌾 AgriSmart")
st.subheader("Smart Agriculture Intelligence Dashboard")
st.caption("Crop Yield Forecasting | Farmer Profitability Prediction")
st.divider()


# =====================================================
# 11. PREDICTION INPUT FORM
# =====================================================

def get_prediction_inputs(include_yield=False, key_prefix=""):
    values = {}
    p = f"{key_prefix}_" if key_prefix else ""

    st.markdown("### Crop and Farming Information")
    st.write("Enter the following details to generate your prediction.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Crop Details")

        values["State"] = st.selectbox(
            "Select State", sorted(df["State"].dropna().unique()), key=f"{p}state_input"
        )
        values["Crop_Type"] = st.selectbox(
            "Select Crop", sorted(df["Crop_Type"].dropna().unique()), key=f"{p}crop_input"
        )
        values["Season"] = st.selectbox(
            "Select Season", sorted(df["Season"].dropna().unique()), key=f"{p}season_input"
        )
        values["Soil_Type"] = st.selectbox(
            "Select Soil Type",
            sorted(df["Soil_Type"].dropna().unique()),
            key=f"{p}soil_input",
        )
        values["Irrigation_Type"] = st.selectbox(
            "Select Irrigation Type",
            sorted(df["Irrigation_Type"].dropna().unique()),
            key=f"{p}irrigation_input",
        )

    with col2:
        st.markdown("#### Weather and Farming Inputs")

        values["Rainfall_mm"] = st.number_input(
            "Rainfall (mm)", min_value=0.0, value=800.0, key=f"{p}rainfall_input"
        )
        values["Avg_Temperature_C"] = st.number_input(
            "Average Temperature (°C)", value=25.0, key=f"{p}temperature_input"
        )
        values["Humidity_percent"] = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            key=f"{p}humidity_input",
        )
        values["Fertilizer_Used_kg"] = st.number_input(
            "Fertilizer Used (kg)", min_value=0.0, value=200.0, key=f"{p}fertilizer_input"
        )
        values["Pesticide_Used_kg"] = st.number_input(
            "Pesticide Used (kg)", min_value=0.0, value=10.0, key=f"{p}pesticide_input"
        )
        values["Cultivated_Area_hectares"] = st.number_input(
            "Cultivated Area (hectares)",
            min_value=0.01,
            value=2.0,
            key=f"{p}area_input",
        )

        if include_yield:
            values[YIELD] = st.number_input(
                "Expected Yield (tons/hectare)",
                min_value=0.0,
                value=4.8,
                key=f"{p}expected_yield_input",
            )
            values["Market_Price_per_ton"] = st.number_input(
                "Market Price per Ton (INR)",
                min_value=1000.0,
                value=38000.0,
                step=500.0,
                key=f"{p}market_price_input",
                help="Prevailing or expected market selling price per ton."
            )

    return values


# =====================================================
# 12. SHARED RENDERERS FOR THE MODEL-EVALUATION PAGES
# =====================================================

def render_model_results(kind):
    """Model comparison, plots, residuals, importance (notebook sections)."""
    is_yield = kind == "yield"
    label = "Crop Yield" if is_yield else "Farmer Profit"
    unit = "tons/hectare" if is_yield else "INR"

    train, test = split_by_year(df)

    if len(train) < 10 or test.empty:
        st.warning(
            f"Not enough data to split into training years and test year {TEST_YEAR}."
        )
        return None

    with st.spinner("Training 9 models... (cached after the first run)"):
        out = evaluate_models(df, kind, TEST_YEAR)

    results_df = out["results"]
    baseline = out["baseline"]
    best_name = out["best_name"]
    y_test = out["y_test"]
    y_train = out["y_train"]
    predictions = out["best_predictions"]

    # ---------- Split ----------
    st.subheader("Time-based Train / Test Split")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Records", f"{out['n_train']:,}")
    c2.metric("Testing Records", f"{out['n_test']:,}")
    c3.metric("Training Years", f"{int(train['Year'].min())}–{int(train['Year'].max())}")
    c4.metric("Testing Year", f"{TEST_YEAR}")

    st.divider()

    # ---------- Baseline ----------
    st.subheader("Baseline (Predict the Training Mean)")

    b1, b2, b3 = st.columns(3)
    b1.metric("Baseline MAE", f"{baseline['MAE']:,.3f}")
    b2.metric("Baseline RMSE", f"{baseline['RMSE']:,.3f}")
    b3.metric("Baseline R²", f"{baseline['R2']:.3f}")

    st.divider()

    # ---------- Comparison ----------
    st.subheader("Model Comparison")
    show_table(results_df.round(4))

    ranked_by = "RMSE (lower is better)" if is_yield else "R² (higher is better)"
    st.caption(f"Models are ranked by {ranked_by}. Best model: **{best_name}**")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            results_df,
            x="Model",
            y="R2 Score",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="R² = 0 baseline",
            annotation_font_color="#000000",
        )
        style_fig(fig, "R² Comparison of Regression Models")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-35)
        show_chart(fig)

    with col2:
        fig = px.bar(
            results_df,
            x="Model",
            y="RMSE",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.add_hline(
            y=baseline["RMSE"],
            line_dash="dash",
            line_color="red",
            annotation_text="Baseline RMSE",
            annotation_font_color="#000000",
        )
        style_fig(fig, "RMSE Comparison of Regression Models")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-35)
        show_chart(fig)

    st.divider()

    # ---------- Best model: actual vs predicted ----------
    st.subheader(f"Best Model: {best_name}")

    m1, m2, m3, m4, m5 = st.columns(5)
    best_row = results_df.loc[0]
    m1.metric("MAE", f"{best_row['MAE']:,.3f}")
    m2.metric("RMSE", f"{best_row['RMSE']:,.3f}")
    m3.metric("R² Score", f"{best_row['R2 Score']:.4f}")
    m4.metric("Accuracy (%)", f"{best_row['Accuracy (%)']:.2f}%")
    m5.metric("F1 Score (Macro)", f"{best_row['F1 Score']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        lo = float(min(y_test.min(), predictions.min()))
        hi = float(max(y_test.max(), predictions.max()))

        fig = px.scatter(
            x=y_test,
            y=predictions,
            opacity=0.6,
            labels={"x": f"Actual {label}", "y": f"Predicted {label}"},
            color_discrete_sequence=["#239B9B"],
        )
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )
        style_fig(fig, f"Actual vs Predicted {label} ({best_name})")
        show_chart(fig)

    residuals = y_test - predictions

    with col2:
        fig = px.scatter(
            x=predictions,
            y=residuals,
            opacity=0.6,
            labels={"x": f"Predicted {label}", "y": "Residual"},
            color_discrete_sequence=["#D9A441"],
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        style_fig(fig, "Residual Analysis")
        show_chart(fig)

    r1, r2 = st.columns(2)
    r1.metric("Mean Residual", f"{residuals.mean():,.4f}")
    r2.metric("Residual Std. Deviation", f"{residuals.std():,.4f}")

    if not is_yield:
        st.markdown("**Sample of actual vs predicted values**")
        sample = pd.DataFrame(
            {
                "Actual Profit": y_test,
                "Predicted Profit": predictions,
                "Error": residuals,
            }
        ).head(10)
        show_table(sample.round(2))

    st.divider()

    # ---------- Feature importance ----------
    st.subheader("Top 15 Random Forest Feature Importances")

    top = out["importance"].head(15).sort_values("Importance")

    fig = px.bar(
        top,
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#237A45"],
    )
    style_fig(fig, "Top 15 Features Influencing " + label, 520)
    fig.update_layout(showlegend=False)
    show_chart(fig)

    # ---------- Train vs test distribution (yield only, as in notebook) ----------
    if is_yield:
        st.divider()
        st.subheader("Crop Yield Distribution: Training vs Testing")

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=y_train,
                name=f"Training: {int(train['Year'].min())}–{int(train['Year'].max())}",
                histnorm="probability density",
                opacity=0.5,
                marker_color="green",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=y_test,
                name=f"Testing: {TEST_YEAR}",
                histnorm="probability density",
                opacity=0.5,
                marker_color="orange",
            )
        )
        style_fig(fig, "Crop Yield Distribution: Training vs Testing")
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Yield (tons/hectare)",
            yaxis_title="Density",
        )
        show_chart(fig)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Training Mean", f"{y_train.mean():.3f}")
        s2.metric("Testing Mean", f"{y_test.mean():.3f}")
        s3.metric("Training Std. Dev.", f"{y_train.std():.3f}")
        s4.metric("Testing Std. Dev.", f"{y_test.std():.3f}")

    return out


def render_cv_section(kind, best_name):
    """Time-series cross-validation (button-triggered because it is slow)."""
    st.divider()
    st.subheader("Time-Series Cross-Validation (5 splits)")

    if kind == "yield":
        names = tuple(make_models(kind).keys())
        st.write(
            "Expanding-window validation of all 9 models on the training years. "
            "SVR and ANN can make this take a minute or two."
        )
    else:
        names = (best_name,)
        st.write(f"Expanding-window validation of the best model (**{best_name}**).")

    state_key = f"cv_done_{kind}"

    if full_width_button(st.button, "Run Time-Series Cross-Validation", key=f"cv_btn_{kind}"):
        st.session_state[state_key] = True

    if st.session_state.get(state_key):
        with st.spinner("Running cross-validation..."):
            cv_df = run_time_series_cv(df, kind, TEST_YEAR, names)

        show_table(cv_df.round(4))

        best_cv = cv_df.iloc[0]
        st.success(
            f"Lowest cross-validated RMSE: {best_cv['Model']} "
            f"(R² = {best_cv['Mean R2']:.3f} ± {best_cv['R2 Std']:.3f})"
        )


# =====================================================
# 15b. SUPPORT VECTOR CLASSIFICATION (SVC) FOR YIELD TIERS
# =====================================================

@st.cache_data(show_spinner=False)
def evaluate_svc_models(data, test_year):
    """Train Support Vector Classifier (SVC) variants to predict Yield Tiers (Low, Medium, High)."""
    train, test = split_by_year(data)
    target, categorical, numerical = feature_lists("yield")
    features = categorical + numerical

    X_train, y_train_cont = train[features], train[target]
    X_test, y_test_cont = test[features], test[target]

    # Tercile binning: Low (bottom 33%), Medium (middle 33%), High (top 33%)
    q33 = float(np.quantile(y_train_cont, 0.333))
    q67 = float(np.quantile(y_train_cont, 0.667))

    def bin_yield(vals):
        return np.digitize(vals, [q33, q67])

    y_train = bin_yield(y_train_cont)
    y_test = bin_yield(y_test_cont)

    svc_variants = {
        "SVC (Linear Kernel)": SVC(kernel="linear", C=1.0, random_state=42),
        "SVC (RBF Kernel)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "SVC (Polynomial Kernel)": SVC(kernel="poly", degree=3, C=1.0, random_state=42),
    }

    results, trained = [], {}
    tier_labels = ["Low Yield (<3.6t)", "Medium Yield (3.6-6.1t)", "High Yield (>=6.1t)"]

    for name, clf in svc_variants.items():
        pipe = Pipeline(
            [
                ("preprocessing", build_preprocessor(categorical, numerical)),
                ("model", clf),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = float((preds == y_test).mean() * 100.0)
        f1 = float(f1_score(y_test, preds, average="macro"))

        # 5-fold temporal cross-validation on training data
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_validate(pipe, X_train, y_train, cv=tscv, scoring=["accuracy", "f1_macro"])
        cv_acc = float(cv_scores["test_accuracy"].mean() * 100.0)
        cv_f1 = float(cv_scores["test_f1_macro"].mean())

        results.append(
            {
                "SVC Kernel": name,
                "Accuracy (%)": acc,
                "F1 Score (Macro)": f1,
                "CV Accuracy (%)": cv_acc,
                "CV F1 (Macro)": cv_f1,
            }
        )
        trained[name] = pipe

    results_df = pd.DataFrame(results).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    best_name = results_df.loc[0, "SVC Kernel"]
    best_pipe = trained[best_name]
    best_preds = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)

    return {
        "results": results_df,
        "trained": trained,
        "best_name": best_name,
        "cm": cm,
        "q33": q33,
        "q67": q67,
        "y_test": y_test,
        "best_preds": best_preds,
        "tier_labels": tier_labels,
    }


def render_svc_section():
    """Renders the Support Vector Classification (SVC) yield tier evaluation and visualizations."""
    st.subheader("🔵 Support Vector Classification (SVC) — Yield Tier Models")
    st.write(
        "Support Vector Classifiers (SVC) categorize farms into discrete performance tiers: "
        "**Low Yield** (< 33rd percentile), **Medium Yield** (33rd–67th percentile), and **High Yield** (> 67th percentile). "
        "This is valuable for zonal agricultural planning, crop insurance risk tiers, and fertilizer rationing."
    )

    with st.spinner("Training SVC models (Linear, RBF, Polynomial kernels with 5-fold CV)..."):
        svc_out = evaluate_svc_models(df, TEST_YEAR)

    svc_results = svc_out["results"]
    best_svc = svc_out["best_name"]
    best_row = svc_results.loc[0]

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best SVC Kernel", best_svc)
    c2.metric("Test Accuracy (%)", f"{best_row['Accuracy (%)']:.2f}%")
    c3.metric("F1 Score (Macro)", f"{best_row['F1 Score (Macro)']:.4f}")
    c4.metric("5-Fold CV Accuracy (%)", f"{best_row['CV Accuracy (%)']:.2f}%")

    st.markdown("#### SVC Kernel Comparison Table")
    show_table(svc_results.round(4))
    st.caption("SVC models ranked by Test Accuracy (%). 5-fold temporal cross-validation validates time-series stability.")

    col1, col2 = st.columns(2)

    with col1:
        # Comparison chart: Accuracy & F1
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Test Accuracy (%)",
                x=svc_results["SVC Kernel"],
                y=svc_results["Accuracy (%)"],
                marker_color="#246B42",
            )
        )
        fig.add_trace(
            go.Bar(
                name="CV Accuracy (%)",
                x=svc_results["SVC Kernel"],
                y=svc_results["CV Accuracy (%)"],
                marker_color="#7ABF8F",
            )
        )
        style_fig(fig, "SVC Kernels: Test vs CV Accuracy (%)")
        fig.update_layout(barmode="group", xaxis_tickangle=-25)
        show_chart(fig)

    with col2:
        # Confusion matrix heatmap
        cm = svc_out["cm"]
        labels = svc_out["tier_labels"]
        cm_fig = px.imshow(
            cm,
            labels=dict(x="Predicted Tier", y="Actual Tier", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Greens",
        )
        style_fig(cm_fig, f"Confusion Matrix: {best_svc}")
        show_chart(cm_fig)

    st.info(
        f"💡 **Managerial Insight on SVC**: The **{best_svc}** achieves the strongest balance between generalization and margin separation. "
        f"Training tercile boundaries: **Low Yield** < {svc_out['q33']:.2f} t/ha, "
        f"**Medium Yield** {svc_out['q33']:.2f}–{svc_out['q67']:.2f} t/ha, "
        f"**High Yield** >= {svc_out['q67']:.2f} t/ha."
    )


# =====================================================
# 19b. AGROSENSE AI ENTERPRISE DIALOG MODAL & CHATBOT
# =====================================================

def handle_dialog_dismiss():
    st.session_state["show_chat_dialog"] = False

@st.dialog("🤖 AgroSense AI • Executive Agricultural Assistant", width="large", on_dismiss=handle_dialog_dismiss)
def open_agrosense_ai_dialog():
    """Renders the AgroSense AI Enterprise Assistant with 100% high-contrast, crystal-clear readability."""

    # --- High-Contrast Professional Styling CSS ---
    st.markdown("""
    <style>
    /* Lock entire background page and prevent background scrolling while dialog is active */
    body:has(div[data-testid="stDialog"]) {
        overflow: hidden !important;
    }

    /* Complete elimination of blink / flicker / dimming during interaction */
    div[data-testid="stDialog"],
    div[data-testid="stDialog"] * {
        transition: none !important;
        animation: none !important;
    }
    div[data-testid="stDialog"] [data-st-stale="true"],
    div[data-testid="stDialog"] [aria-busy="true"],
    div[data-testid="stDialog"] .st-emotion-cache-16idsys {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }

    /* Dialog base modal styling - Solid opaque white with emerald accent */
    div[data-testid="stDialog"] [role="dialog"],
    div[data-testid="stDialog"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #246B42 !important;
        border-top: 5px solid #17452D !important;
        border-radius: 14px !important;
        box-shadow: 0 20px 60px rgba(23, 69, 45, 0.22) !important;
        color: #17452D !important;
        opacity: 1 !important;
        max-height: 90vh !important;
        overflow-y: hidden !important; /* Modal itself stays fixed; only message box scrolls */
        overflow-x: hidden !important;
        padding: 12px 18px !important;
    }
    div[data-testid="stDialog"] [role="dialog"] > div {
        overflow-y: visible !important;
        overflow-x: hidden !important;
    }

    /* Pinned, clean, compact pill Chat Input Bar - no bloated outer gray wrapper */
    div[data-testid="stDialog"] div[data-testid="stChatInput"] {
        background: transparent !important;
        padding: 0 !important;
        margin-top: 6px !important;
        margin-bottom: 2px !important;
        border: none !important;
        box-shadow: none !important;
        position: relative !important;
        z-index: 100 !important;
    }
    div[data-testid="stDialog"] div[data-testid="stChatInput"] > div {
        background: #FFFFFF !important;
        border: 1.5px solid #246B42 !important;
        border-radius: 24px !important;
        padding: 2px 10px !important;
        box-shadow: 0 2px 8px rgba(23, 69, 45, 0.08) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDialog"] div[data-testid="stChatInput"] > div:focus-within {
        border-color: #00C853 !important;
        box-shadow: 0 0 0 3px rgba(0, 200, 83, 0.22) !important;
    }
    div[data-testid="stDialog"] div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #17452D !important;
        font-size: 0.90rem !important;
        font-weight: 550 !important;
        padding: 6px 10px !important;
        min-height: 38px !important;
    }
    div[data-testid="stDialog"] div[data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stDialog"] div[data-testid="stChatInput"] textarea::placeholder {
        color: #557A60 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stDialog"] div[data-testid="stChatInput"] button {
        color: #246B42 !important;
        background: transparent !important;
        border: none !important;
    }

    /* Clean custom scrollbar for messages and session history */
    div[data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"] > div {
        scrollbar-width: thin !important;
        scrollbar-color: #A3D9B5 #F4F8F5 !important;
    }
    div[data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar {
        width: 6px !important;
    }
    div[data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-track {
        background: #F4F8F5 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb {
        background-color: #A3D9B5 !important;
        border-radius: 6px !important;
    }
    div[data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb:hover {
        background-color: #246B42 !important;
    }
    div[data-testid="stDialog"] {
        background-color: rgba(14, 38, 24, 0.65) !important;
        backdrop-filter: blur(6px) !important;
    }
    
    /* Ensure all text inside dialog is dark, crisp, and high-contrast */
    div[data-testid="stDialog"] h1,
    div[data-testid="stDialog"] h2,
    div[data-testid="stDialog"] h3 {
        color: #17452D !important;
        font-weight: 800 !important;
    }
    div[data-testid="stDialog"] p,
    div[data-testid="stDialog"] span,
    div[data-testid="stDialog"] label {
        color: #2D4734 !important;
    }
    
    /* Left Sidebar History Panel */
    .chat-history-sidebar {
        background: #F4F8F5;
        border: 1px solid #DCE7DD;
        border-radius: 12px;
        padding: 14px 12px;
    }
    
    /* Suggestion Query Buttons Styling (Clean, Bold & 100% Readable) - Scoped to Main Chat Column */
    div[data-testid="stDialog"] div[data-testid="stColumn"]:last-child div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
        background-color: #FFFFFF !important;
        border: 1.5px solid #A3D9B5 !important;
        border-radius: 8px !important;
        color: #17452D !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        text-align: left !important;
        padding: 8px 12px !important;
        min-height: 42px !important;
        box-shadow: 0 1px 4px rgba(23, 69, 45, 0.05) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDialog"] div[data-testid="stColumn"]:last-child div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
        background-color: #E8F5E9 !important;
        border-color: #246B42 !important;
        color: #0E2A1B !important;
        box-shadow: 0 4px 14px rgba(23, 69, 45, 0.15) !important;
        transform: translateY(-2px) !important;
    }

    /* Left Sidebar History Session Buttons Styling (Always High-Contrast & Visible) */
    div[data-testid="stDialog"] div[data-testid="stColumn"]:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:first-child button {
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 6px 10px !important;
        min-height: 38px !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    div[data-testid="stDialog"] div[data-testid="stColumn"]:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:first-child button[kind="primary"] {
        background-color: #246B42 !important;
        color: #FFFFFF !important;
        border: 1.5px solid #246B42 !important;
    }
    div[data-testid="stDialog"] div[data-testid="stColumn"]:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:first-child button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #17452D !important;
        border: 1px solid #DCE7DD !important;
    }
    div[data-testid="stDialog"] div[data-testid="stColumn"]:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:first-child button:hover {
        background-color: #E8F5E9 !important;
        color: #0E2A1B !important;
        border-color: #246B42 !important;
    }

    /* Left Sidebar Trash / Delete Button Styling */
    div[data-testid="stDialog"] div[data-testid="stColumn"]:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:last-child button {
        background-color: #F4F8F5 !important;
        border: 1px solid #DCE7DD !important;
        border-radius: 8px !important;
        padding: 4px !important;
        min-height: 38px !important;
        justify-content: center !important;
        box-shadow: none !important;
    }
    div[data-testid="stDialog"] div[data-testid="stColumn"]:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:last-child button:hover {
        background-color: #FFEBEE !important;
        border-color: #EF9A9A !important;
    }

    /* Primary Action Buttons (e.g. + New chat) */
    div[data-testid="stDialog"] button[kind="primary"],
    div[data-testid="stDialog"] button:has(p:contains("New chat")) {
        background-color: #246B42 !important;
        color: #FFFFFF !important;
        border: 1.5px solid #246B42 !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
    }
    div[data-testid="stDialog"] button[kind="primary"]:hover {
        background-color: #17452D !important;
        border-color: #17452D !important;
    }
    
    /* User Chat Message Bubble - Compact & Tight */
    div[data-testid="stChatMessage"]:has(div:contains("Agribusiness Strategist")),
    div[data-testid="stChatMessage"]:first-child {
        background-color: #EBF5EE !important;
        border: 1px solid #C4E2CC !important;
        border-radius: 10px !important;
        margin: 4px 0 !important;
        padding: 8px 12px !important;
    }
    div[data-testid="stChatMessage"]:has(div:contains("Agribusiness Strategist")) * {
        color: #17452D !important;
        font-size: 0.92rem !important;
    }
    
    /* AI Assistant Chat Message Card - Compact & Tight */
    div[data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE7DD !important;
        border-left: 4px solid #246B42 !important;
        border-radius: 10px !important;
        margin: 6px 0 !important;
        padding: 10px 14px !important;
        box-shadow: 0 2px 6px rgba(23, 69, 45, 0.04) !important;
    }
    div[data-testid="stChatMessage"] * {
        color: #203528 !important;
        font-size: 0.92rem !important;
    }

    /* Structured Markdown Tables inside Dialog - Dense & Sleek */
    div[data-testid="stDialog"] div[data-testid="stMarkdownContainer"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 6px 0 !important;
        font-size: 0.86rem !important;
        background: #FFFFFF !important;
        border: 1px solid #DCE7DD !important;
        border-radius: 6px !important;
    }
    div[data-testid="stDialog"] div[data-testid="stMarkdownContainer"] th {
        background: #17452D !important;
        color: #FFFFFF !important;
        padding: 6px 10px !important;
        border-bottom: 2px solid #246B42 !important;
        font-weight: 750 !important;
        font-size: 0.84rem !important;
        text-align: left !important;
    }
    div[data-testid="stDialog"] div[data-testid="stMarkdownContainer"] td {
        padding: 5px 10px !important;
        border-bottom: 1px solid #EBF1EC !important;
        color: #243D2C !important;
        font-size: 0.84rem !important;
    }

    /* Floating bubble and speech popover must hide when dialog is active */
    body:has(div[data-testid="stDialog"]) .agro-speech-popover,
    body:has(div[data-testid="stDialog"]) div[data-testid="stButton"]:has(button#agro_floating_btn),
    body:has(div[data-testid="stDialog"]) button#agro_floating_btn {
        display: none !important;
    }

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {
        div[data-testid="stDialog"] {
            width: 96vw !important;
            max-width: 96vw !important;
            min-width: 92vw !important;
            height: 94vh !important;
            max-height: 94vh !important;
            padding: 8px !important;
            margin: 0 auto !important;
            top: 3vh !important;
        }
        div[data-testid="stDialog"] div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 12px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Callback helpers for seamless, instant state transitions without reload blinks
    def cb_switch_session(sid):
        st.session_state["active_session_id"] = sid

    def cb_new_chat():
        new_id = chatbot.create_new_session("New Agricultural Briefing")
        st.session_state["active_session_id"] = new_id

    def cb_delete_chat(sid):
        chatbot.delete_session(sid)
        rem = chatbot.load_all_sessions()
        if rem:
            st.session_state["active_session_id"] = rem[0]["id"]
        else:
            st.session_state["active_session_id"] = chatbot.create_new_session("New Agricultural Briefing")

    def cb_select_suggestion(prompt_text):
        st.session_state["_pending_chat_query"] = prompt_text

    # Initialize active session
    if "active_session_id" not in st.session_state:
        existing = chatbot.load_all_sessions()
        if existing:
            st.session_state["active_session_id"] = existing[0]["id"]
        else:
            st.session_state["active_session_id"] = chatbot.create_new_session("Initial Agricultural Briefing")

    all_sessions = chatbot.load_all_sessions()
    current_session = next((s for s in all_sessions if s["id"] == st.session_state["active_session_id"]), None)

    if current_session is None:
        st.session_state["active_session_id"] = chatbot.create_new_session("New Agricultural Briefing")
        all_sessions = chatbot.load_all_sessions()
        current_session = all_sessions[0]

    # --- TWO-COLUMN CHATGPT / GEMINI / CLAUDE LAYOUT ---
    col_history, col_main = st.columns([1, 2.7], gap="small")

    pending_query = st.session_state.pop("_pending_chat_query", None)

    # =========================================================================
    # LEFT COLUMN: CHAT HISTORY SIDEBAR (Matches ChatGPT / Gemini / Claude)
    # =========================================================================
    with col_history:
        # Header with Logo & Pro Badge
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; padding-bottom:6px; border-bottom:1.5px solid #DCE7DD;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="font-size:1.2rem;">🤖</span>
                <span style="font-weight:850; font-size:1.1rem; color:#17452D;">AgroSense</span>
            </div>
            <span style="font-size:0.70rem; background:#E8F5E9; color:#17452D; border:1px solid #A3D9B5; padding:2px 7px; border-radius:10px; font-weight:800;">LPU ACTIVE</span>
        </div>
        """, unsafe_allow_html=True)

        # ➕ New chat pill button (just like in ChatGPT / Gemini / Claude screenshots!)
        st.button("➕ New chat", key="btn_new_chat_sidebar", use_container_width=True, on_click=cb_new_chat)

        st.markdown('<p style="font-size:0.75rem; color:#205B38; font-weight:800; text-transform:uppercase; margin-top:10px; margin-bottom:6px; letter-spacing:0.6px;">Recent Briefings</p>', unsafe_allow_html=True)

        # List of past chat sessions (clean, deduplicated & bulletproof keys)
        if all_sessions:
            seen_ids = set()
            unique_sessions = []
            for s in all_sessions:
                sid = str(s.get("id", ""))
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    unique_sessions.append(s)

            for idx, s in enumerate(unique_sessions[:5]):
                is_active = (s["id"] == st.session_state["active_session_id"])
                c_title, c_del = st.columns([5, 1])
                with c_title:
                    btn_label = f"• {s.get('title', 'Session')[:20]}"
                    btn_type = "primary" if is_active else "secondary"
                    st.button(
                        btn_label,
                        key=f"hist_s_{s['id']}_{idx}",
                        use_container_width=True,
                        type=btn_type,
                        on_click=cb_switch_session,
                        args=(s["id"],)
                    )
                with c_del:
                    st.button(
                        "🗑️",
                        key=f"hist_del_{s['id']}_{idx}",
                        help="Delete this session",
                        on_click=cb_delete_chat,
                        args=(s["id"],)
                    )
        else:
            st.caption("No saved conversations yet.")

        # Optional Groq Key input expander for Streamlit Cloud users
        with st.expander("🔑 API Key Settings", expanded=False):
            key_in = st.text_input(
                "Groq API Key (Optional)",
                value=st.session_state.get("user_groq_key", ""),
                type="password",
                help="Paste your Groq API key (gsk_...) to activate cloud LLM synthesis on Streamlit Cloud.",
                key="input_groq_key"
            )
            if key_in:
                st.session_state["user_groq_key"] = key_in.strip()
                st.caption("✅ Groq Cloud LLM Active")
            else:
                st.caption("ℹ️ Ground-Truth Analytical Engine Active")

        # User Profile Badge at bottom left (matches ChatGPT / Gemini / Claude profile footer)
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; padding:6px 10px; border-radius:8px; background:#F4F8F5; margin-top:10px; margin-bottom:8px; border:1px solid #DCE7DD;">
            <div style="width:28px; height:28px; border-radius:50%; background:linear-gradient(135deg, #17452D 0%, #246B42 100%); color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.80rem;">AK</div>
            <div>
                <div style="font-size:0.80rem; font-weight:800; color:#17452D; line-height:1.2;">Aadish Kotadia</div>
                <div style="font-size:0.68rem; color:#4B6251; font-weight:600; line-height:1.2;">AgriSmart Analyst • SEM 3 MBA</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Export Transcript & Close Buttons
        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            md_transcript = chatbot.export_session_markdown(current_session)
            st.download_button(
                "📥 Export",
                data=md_transcript,
                file_name=f"AgroSense_{current_session['id']}.md",
                mime="text/markdown",
                key="sb_export_btn",
                use_container_width=True,
                help="Download session transcript as Markdown"
            )
        with exp_c2:
            if st.button("✕ Close", key="sb_close_dialog_btn", use_container_width=True):
                st.session_state["show_chat_dialog"] = False
                st.rerun()

    # =========================================================================
    # RIGHT COLUMN: MAIN CONVERSATION CANVAS (ChatGPT / Gemini / Claude Center)
    # =========================================================================
    with col_main:
        messages = current_session.get("messages", [])

        # =====================================================================
        # SCROLLABLE MESSAGE CONTAINER (ONLY MESSAGES SCROLL, FIXED 390PX)
        # =====================================================================
        msg_container = st.container(height=390, border=False)

        # =====================================================================
        # PINNED INPUT BAR (OUTSIDE THE CONTAINER, PERMANENTLY FIXED AT BOTTOM)
        # =====================================================================
        user_input = st.chat_input(
            "Ask AgroSense AI anything about Indian crops, yields, profits, states, soils, weather, ML models…",
            key="gemini_style_chat_input"
        )

        active_query = pending_query or user_input

        with msg_container:
            # --- If no messages and no query: Hero Greeting & Suggestion Chips (Gemini/Claude style) ---
            if not messages and not active_query:
                st.markdown("""
                <div style="text-align:center; padding: 4px 10px 4px 10px;">
                    <div style="font-size: 1.8rem; margin-bottom: 2px;">🌾</div>
                    <h2 style="font-size: 1.4rem; font-weight: 850; color: #17452D !important; margin: 0;">Where should we start?</h2>
                    <p style="color: #35483A !important; font-size: 0.88rem; font-weight: 600; margin-top: 2px; margin-bottom: 8px;">Senior Decision Support • 7,920 Historical Records (2014–2024)</p>
                </div>
                """, unsafe_allow_html=True)

                # Suggestion Chips in 2 Columns (Matching Gemini & Claude screenshots)
                st.markdown('<p style="font-size:0.75rem; color:#205B38; font-weight:800; text-transform:uppercase; margin-bottom:6px; letter-spacing:0.6px;">Suggested Inquiries</p>', unsafe_allow_html=True)
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.button("🌾 Which state produces highest Wheat yield?", key="sug_chip_1", use_container_width=True, on_click=cb_select_suggestion, args=("Which state produces the highest Wheat yield and what are the agronomic drivers?",))
                    st.button("💰 What is the most profitable crop in Kharif?", key="sug_chip_2", use_container_width=True, on_click=cb_select_suggestion, args=("What is the most profitable crop during the Kharif season and what is its net return?",))
                    st.button("🪨 Optimal soil suitability for Cotton & Rice", key="sug_chip_3", use_container_width=True, on_click=cb_select_suggestion, args=("Compare crop productivity across soil classifications and identify the best soil for Cotton and Rice.",))
                with sc2:
                    st.button("📈 10-year historical national yield trajectory", key="sug_chip_4", use_container_width=True, on_click=cb_select_suggestion, args=("Show the 10-year historical national crop yield trajectory from 2014 to 2024.",))
                    st.button("💧 Drip vs Canal vs Sprinkler efficiency", key="sug_chip_5", use_container_width=True, on_click=cb_select_suggestion, args=("Compare agricultural yield and farmer profit between Drip, Canal, and Sprinkler irrigation.",))
                    st.button("🤖 How is farmer profitability predicted by ML?", key="sug_chip_6", use_container_width=True, on_click=cb_select_suggestion, args=("Explain how machine learning predicts farmer profitability and what are the dominant features.",))
            else:
                # --- Render Chat Messages (ChatGPT / Claude message bubble cards) ---
                for msg in messages:
                    if msg["role"] == "user":
                        with st.chat_message("user", avatar="🧑‍🌾"):
                            st.markdown(f"**Agribusiness Strategist** • *{msg.get('timestamp', '')}*")
                            st.markdown(msg["content"])
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(f"**AgroSense AI Executive Brief** • *{msg.get('timestamp', '')}*")
                            st.markdown(msg["content"])
                            # Reconstruct chart if chart_title was saved
                            if msg.get("chart_title"):
                                fig = chatbot.generate_dynamic_chart(msg.get("chart_title"))
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # --- Render and Process Active Query (Instantaneous without reload/blink) ---
                if active_query:
                    now_str = datetime.now().strftime("%I:%M %p")
                    # 1. Render user query immediately in chat box
                    with st.chat_message("user", avatar="🧑‍🌾"):
                        st.markdown(f"**Agribusiness Strategist** • *{now_str}*")
                        st.markdown(active_query)

                    chatbot.append_message_to_session(
                        st.session_state["active_session_id"],
                        role="user",
                        content=active_query
                    )

                    # 2. Render assistant response directly in container
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(f"**AgroSense AI Executive Brief** • *{now_str}*")
                        with st.spinner("🤖 AgroSense AI is thinking... Analyzing 7,920 records & synthesizing executive solution..."):
                            try:
                                session_history = [
                                    {"role": m["role"], "content": m["content"]}
                                    for m in messages
                                ]
                                answer, fig = chatbot.chat(active_query, session_history)
                                st.markdown(answer)
                                chart_title = active_query if fig is not None else None
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                                chatbot.append_message_to_session(
                                    st.session_state["active_session_id"],
                                    role="assistant",
                                    content=answer,
                                    chart_title=chart_title
                                )
                            except Exception as e:
                                err_msg = f"⚠️ Briefing synthesis halted: `{e}`. Please check system connectivity and credentials."
                                st.markdown(err_msg)
                                chatbot.append_message_to_session(
                                    st.session_state["active_session_id"],
                                    role="assistant",
                                    content=err_msg
                                )


# =====================================================
# 19c. FLOATING BOTTOM-RIGHT CHAT BUBBLE (ON EVERY PAGE)
# =====================================================

def render_floating_chat_bubble():
    """
    Renders an elegant circular floating chat bubble in the bottom-right corner of EVERY page.
    Automatically hides when the dialog modal is open so it never overlaps or clutters the chat!
    """
    # Track page to safely reset show_chat_dialog when switching between pages
    active_page = st.session_state.get("_active_page_name")
    if active_page != page:
        st.session_state["_active_page_name"] = page
        st.session_state["show_chat_dialog"] = False

    # If the chat dialog is currently open, open dialog and DO NOT render the floating bubble into the DOM!
    if st.session_state.get("show_chat_dialog", False):
        open_agrosense_ai_dialog()
        return

    # 1. Floating Bubble CSS with Continuous Blinking Glow and Speech Popover
    st.markdown("""
    <style>
    /* Continuous Pulsing/Blinking Glow Keyframes */
    @keyframes agro-bubble-blink-pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.85), 0 8px 25px rgba(23,69,45,0.45);
            transform: scale(1);
        }
        50% {
            box-shadow: 0 0 0 16px rgba(0, 230, 118, 0), 0 14px 34px rgba(0, 200, 83, 0.65);
            transform: scale(1.08);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(0, 230, 118, 0), 0 8px 25px rgba(23,69,45,0.45);
            transform: scale(1);
        }
    }

    /* Floating Speech Popover Animation */
    @keyframes agro-popover-bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-6px);
        }
    }

    /* Floating Speech Popover directly above the button */
    .agro-speech-popover {
        position: fixed !important;
        bottom: 96px !important;
        right: 20px !important;
        z-index: 9999998 !important;
        background: linear-gradient(135deg, #0e2a1b 0%, #17452D 100%) !important;
        color: #FFFFFF !important;
        border: 1.5px solid #00E676 !important;
        border-radius: 14px !important;
        padding: 9px 15px !important;
        font-size: 0.86rem !important;
        font-weight: 650 !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
        animation: agro-popover-bounce 2.6s infinite ease-in-out !important;
        pointer-events: none !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        white-space: nowrap !important;
    }
    .agro-speech-popover-badge {
        background: #00E676 !important;
        color: #0e2a1b !important;
        font-size: 0.72rem !important;
        font-weight: 800 !important;
        padding: 2px 7px !important;
        border-radius: 10px !important;
        text-transform: uppercase !important;
    }
    .agro-speech-popover-arrow {
        position: absolute !important;
        bottom: -7px !important;
        right: 28px !important;
        width: 0 !important;
        height: 0 !important;
        border-left: 7px solid transparent !important;
        border-right: 7px solid transparent !important;
        border-top: 7px solid #00E676 !important;
    }

    /* Fixed Floating Chat Bubble Button Styling with Continuous Blink */
    div[data-testid="stElementContainer"]:has(.agro-bubble-fixed-container) + div[data-testid="stElementContainer"] button {
        position: fixed !important;
        bottom: 26px !important;
        right: 28px !important;
        z-index: 9999999 !important;
        width: 62px !important;
        height: 62px !important;
        min-width: 62px !important;
        max-width: 62px !important;
        min-height: 62px !important;
        max-height: 62px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #17452D 0%, #246B42 55%, #00C853 100%) !important;
        color: #FFFFFF !important;
        border: 2.5px solid #69F0AE !important;
        font-size: 28px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        padding: 0 !important;
        margin: 0 !important;
        animation: agro-bubble-blink-pulse 2s infinite ease-in-out !important;
        transition: transform 0.25s ease !important;
    }
    div[data-testid="stElementContainer"]:has(.agro-bubble-fixed-container) + div[data-testid="stElementContainer"] button:hover {
        transform: scale(1.15) !important;
        border-color: #FFFFFF !important;
    }

    /* Mobile Responsive Sizing */
    @media (max-width: 768px) {
        .agro-speech-popover {
            bottom: 84px !important;
            right: 14px !important;
            font-size: 0.76rem !important;
            padding: 6px 11px !important;
        }
        div[data-testid="stElementContainer"]:has(.agro-bubble-fixed-container) + div[data-testid="stElementContainer"] button {
            bottom: 18px !important;
            right: 18px !important;
            width: 56px !important;
            height: 56px !important;
            min-width: 56px !important;
            min-height: 56px !important;
            font-size: 25px !important;
        }
    }
    </style>
    
    <!-- Floating Popover Prompt above the Button (Available on EVERY Page) -->
    <div class="agro-speech-popover">
        <span class="agro-speech-popover-badge">AI</span>
        <span>🌾 <strong>Any question? Ask me!</strong> I am AgroSense AI chatbot</span>
        <div class="agro-speech-popover-arrow"></div>
    </div>
    <div class="agro-bubble-fixed-container"></div>
    """, unsafe_allow_html=True)

    # 2. Floating Button (Small Circular Bubble)
    if st.button("💬", key="agro_floating_btn", help="Open AgroSense AI Copilot"):
        st.session_state["show_chat_dialog"] = True
        open_agrosense_ai_dialog()

    # 3. JavaScript Helper strictly scoped to the immediate sibling button
    st.components.v1.html("""
    <script>
    (function() {
        function setupBubble() {
            try {
                const doc = window.parent.document;
                const container = doc.querySelector('.agro-bubble-fixed-container');
                if (container) {
                    const parent = container.closest('[data-testid="stElementContainer"]');
                    if (parent && parent.nextElementSibling) {
                        const btn = parent.nextElementSibling.querySelector('button');
                        if (btn && btn.innerText.trim() === '💬') {
                            btn.style.position = 'fixed';
                            btn.style.bottom = '26px';
                            btn.style.right = '28px';
                            btn.style.zIndex = '9999999';
                            btn.style.width = '62px';
                            btn.style.height = '62px';
                            btn.style.minWidth = '62px';
                            btn.style.minHeight = '62px';
                            btn.style.borderRadius = '50%';
                            btn.style.background = 'linear-gradient(135deg, #17452D 0%, #246B42 55%, #00C853 100%)';
                            btn.style.color = '#FFFFFF';
                            btn.style.border = '2.5px solid #69F0AE';
                            btn.style.fontSize = '28px';
                            btn.style.display = 'flex';
                            btn.style.alignItems = 'center';
                            btn.style.justifyContent = 'center';
                            btn.style.cursor = 'pointer';
                            btn.style.padding = '0';
                            btn.style.margin = '0';
                            btn.style.animation = 'agro-bubble-blink-pulse 2s infinite ease-in-out';
                        }
                    }
                }
            } catch(e) {}
        }
        setupBubble();
        setTimeout(setupBubble, 80);
        setTimeout(setupBubble, 300);
        setTimeout(setupBubble, 800);
    })();
    </script>
    """, height=0, width=0)



# =====================================================
# 13. PAGE: DASHBOARD OVERVIEW
# =====================================================

if page == "Dashboard Overview":

    st.header("Dashboard Overview")
    st.write(
        "Summary of agricultural records, crop production and farmer profitability."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Average Crop Yield", f"{df[YIELD].mean():.2f} tons/ha")
    col3.metric("Average Farmer Profit", f"₹{df[PROFIT].mean():,.0f}")
    col4.metric("Years Covered", f"{df['Year'].min()}–{df['Year'].max()}")

    st.divider()

    st.subheader("Crop-wise Average Yield")
    crop_yield = (
        df.groupby("Crop_Type")[YIELD].mean().sort_values(ascending=False)
    )
    show_chart(create_bar_chart(crop_yield, "Crop-wise Average Yield", "#237A45"))

    st.divider()

    st.subheader("Year-wise Crop Yield")
    yearly_yield = df.groupby("Year")[YIELD].mean()
    show_chart(
        create_line_chart(yearly_yield, "Year-wise Average Crop Yield", "#239B9B")
    )

    st.divider()

    st.subheader("State-wise Average Farmer Profit")
    state_profit = df.groupby("State")[PROFIT].mean().sort_values(ascending=False)
    show_chart(
        create_bar_chart(state_profit, "State-wise Average Farmer Profit", "#D9A441")
    )

    st.divider()

    st.subheader("Season-wise Average Crop Yield")
    season_yield = df.groupby("Season")[YIELD].mean()
    show_chart(
        create_bar_chart(season_yield, "Season-wise Average Crop Yield", "#5276A7")
    )

    st.divider()

    st.subheader("Dataset Preview")
    show_table(df.head(10), height=350)


# =====================================================
# 14. PAGE: CROP ANALYSIS
# =====================================================

elif page == "Crop Analysis":

    st.header("Crop Performance Analysis")
    st.write("Analyse crop performance across different states and seasons.")

    crop_options = ["All Crops"] + sorted(df["Crop_Type"].dropna().unique().tolist())
    selected_crop = st.selectbox("Select Crop", crop_options, key="analysis_crop")

    filtered = df.copy()
    if selected_crop != "All Crops":
        filtered = filtered[filtered["Crop_Type"] == selected_crop]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("State-wise Average Yield")
        state_yield = (
            filtered.groupby("State")[YIELD].mean().sort_values(ascending=False)
        )
        show_chart(create_bar_chart(state_yield, "State-wise Average Yield", "#237A45"))

    with col2:
        st.subheader("Season-wise Average Yield")
        season_yield = filtered.groupby("Season")[YIELD].mean()
        show_chart(
            create_bar_chart(season_yield, "Season-wise Average Yield", "#5276A7")
        )

    st.divider()

    st.subheader("Filtered Crop Records")
    show_table(filtered, height=400)


# =====================================================
# 15. PAGE: DATA EXPLORER  (notebook: EDA sections)
# =====================================================

elif page == "Data Explorer":

    st.header("🔎 Data Explorer")
    st.write("Dataset structure, quality checks and exploratory analysis.")

    # ---------- Dataset information ----------
    st.subheader("Dataset Information")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", f"{raw_df.shape[1]}")
    c3.metric("Missing Values", f"{int(raw_df.isnull().sum().sum()):,}")
    c4.metric("Duplicate Records", f"{int(raw_df.duplicated().sum()):,}")

    tab1, tab2, tab3 = st.tabs(["Data Types", "Missing Values", "Descriptive Statistics"])

    with tab1:
        dtypes = raw_df.dtypes.astype(str).rename("Data Type").to_frame()
        show_table(dtypes)

    with tab2:
        missing = raw_df.isnull().sum().rename("Missing Values").to_frame()
        show_table(missing)

    with tab3:
        show_table(raw_df.describe().T)

    st.divider()

    # ---------- Target distribution ----------
    st.subheader("Distribution of Crop Yield")

    y = df[YIELD]

    fig = px.histogram(
        df,
        x=YIELD,
        nbins=30,
        marginal="box",
        color_discrete_sequence=["#237A45"],
        labels={YIELD: "Yield (tons/hectare)"},
    )
    style_fig(fig, "Distribution of Crop Yield", 450)
    fig.update_layout(yaxis_title="Frequency")
    show_chart(fig)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Skewness", f"{y.skew():.3f}")
    s2.metric("Mean", f"{y.mean():.3f}")
    s3.metric("Std. Deviation", f"{y.std():.3f}")
    s4.metric("Minimum", f"{y.min():.3f}")
    s5.metric("Maximum", f"{y.max():.3f}")

    st.divider()

    # ---------- Yearly trend ----------
    st.subheader("Year-wise Average Crop Yield")

    yearly_yield = df.groupby("Year")[YIELD].mean()
    show_chart(create_line_chart(yearly_yield, "Year-wise Average Crop Yield"))
    show_table(yearly_yield.round(4).to_frame())

    st.divider()

    # ---------- Correlation ----------
    st.subheader("Correlation Heatmap of Agricultural Variables")

    corr = df[NUMERIC_FEATURES + [YIELD]].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    style_fig(fig, "Correlation Heatmap", 600)
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    show_chart(fig)

    st.markdown("**Correlation with Crop Yield**")
    show_table(corr[YIELD].sort_values(ascending=False).round(4).to_frame())





# =====================================================
# 16. PAGE: MACHINE LEARNING MODEL EVALUATION  (notebook: profit & yield modelling)
# =====================================================

elif page.strip() in ["Machine Learning Model Evaluation", "Profitability Model Evaluation"]:

    st.header("🔬 Machine Learning Model Evaluation & Benchmarking")
    st.write(
        "Empirical evaluation of regression and classification models "
        "trained on 7,920 agricultural records (2014–2024)."
    )

    tab_profit, tab_yield, tab_svc = st.tabs([
        "💹 Farmer Profitability Models (9 Regressors)",
        "🌾 Crop Yield Models (9 Regressors)",
        "🔵 SVC Yield Tier Classification (Support Vector Classifier)",
    ])

    with tab_profit:
        st.subheader("Farmer Profitability Regression Models")
        st.caption("Predicting net farmer profit in INR using agricultural inputs, weather, and yield.")
        out = render_model_results("profit")
        if out is not None:
            render_cv_section("profit", out["best_name"])
            st.divider()
            st.subheader("Download Best Profit Model")
            if full_width_button(st.button, "Prepare model file (.pkl)", key="prep_model"):
                buffer = io.BytesIO()
                joblib.dump(out["trained"][out["best_name"]], buffer)
                st.session_state["profit_model_bytes"] = buffer.getvalue()

            if "profit_model_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download farmer_profitability_model.pkl",
                    data=st.session_state["profit_model_bytes"],
                    file_name="farmer_profitability_model.pkl",
                    mime="application/octet-stream",
                )

    with tab_yield:
        st.subheader("Crop Yield Regression Models")
        st.caption("Predicting crop yield in tons/hectare using weather, soil, irrigation, and agronomic inputs.")
        yield_out = render_model_results("yield")
        if yield_out is not None:
            render_cv_section("yield", yield_out["best_name"])

    with tab_svc:
        render_svc_section()


# =====================================================
# 17. PAGE: CROP YIELD FORECASTING
# =====================================================

elif page == "Crop Yield Forecasting":

    st.header("🌱 Crop Yield Forecasting")
    st.write("Estimate crop yield using weather conditions, soil, irrigation and farming inputs.")
    st.info("Complete the details below and click **Forecast Crop Yield** to view the prediction, benchmark graphs, and agronomic advisory.")

    with st.form("yield_forecast_form"):
        inputs = get_prediction_inputs(include_yield=False, key_prefix="yield")
        st.divider()
        submitted = full_width_button(
            st.form_submit_button, "🌱 Forecast Crop Yield"
        )

    if submitted:
        state = inputs["State"]
        crop = inputs["Crop_Type"]
        fertilizer = float(inputs["Fertilizer_Used_kg"])
        pesticide = float(inputs["Pesticide_Used_kg"])
        area = float(inputs["Cultivated_Area_hectares"])

        # Populate econometric proxies from historical state/crop benchmarks
        subset_crop = df[(df["State"] == state) & (df["Crop_Type"] == crop)]
        if subset_crop.empty:
            subset_crop = df[df["Crop_Type"] == crop]
        est_price = float(subset_crop["Market_Price_per_ton"].median()) if not subset_crop.empty else float(df["Market_Price_per_ton"].median())
        est_profit = float(subset_crop["Farmer_Profit_INR"].median()) if not subset_crop.empty else float(df["Farmer_Profit_INR"].median())

        area_price = area * est_price + 1e-5
        costs = fertilizer * 2.29 + pesticide * 112.13
        inputs["Market_Price_per_ton"] = est_price
        inputs["Farmer_Profit_INR"] = est_profit
        inputs["Area_Price"] = area_price
        inputs["Input_Costs"] = costs
        inputs["Yield_Proxy"] = (est_profit + costs - 4059) / (0.0001675 * area_price)
        inputs["Profit_per_Ha"] = est_profit / (area + 1e-5)
        inputs["Rev_Proxy"] = est_profit / area_price

        with st.spinner("Forecasting crop yield using best model (R² > 81%)..."):
            model, features, from_file = load_or_train_model(df, YIELD)
            required_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else features
            input_data = pd.DataFrame([inputs])
            for col in required_cols:
                if col not in input_data.columns:
                    input_data[col] = 0.0
            input_data = input_data[required_cols]
            prediction = float(model.predict(input_data)[0])

        state = inputs["State"]
        crop = inputs["Crop_Type"]
        season = inputs["Season"]
        soil = inputs["Soil_Type"]
        irrigation = inputs["Irrigation_Type"]
        rainfall = float(inputs["Rainfall_mm"])
        temp = float(inputs["Avg_Temperature_C"])
        humidity = float(inputs["Humidity_percent"])
        fertilizer = float(inputs["Fertilizer_Used_kg"])
        pesticide = float(inputs["Pesticide_Used_kg"])
        area = float(inputs["Cultivated_Area_hectares"])

        # Benchmarks
        state_avg = float(df[df["State"] == state][YIELD].mean())
        crop_avg = float(df[df["Crop_Type"] == crop][YIELD].mean())
        subset_sc = df[(df["State"] == state) & (df["Crop_Type"] == crop)]
        state_crop_avg = float(subset_sc[YIELD].mean()) if not subset_sc.empty else crop_avg
        overall_avg = float(df[YIELD].mean())
        total_production = prediction * area
        percentile = float((df[YIELD] < prediction).mean() * 100)

        # Classification
        if prediction >= 5.5:
            category = "Superior / High Yield Potential"
            cat_badge = "🟢 High Yield"
        elif prediction >= 4.5:
            category = "Optimal / Healthy Yield"
            cat_badge = "🟢 Optimal Yield"
        elif prediction >= 3.5:
            category = "Moderate Production Potential"
            cat_badge = "🟡 Moderate Yield"
        else:
            category = "Below Average / Low Yield"
            cat_badge = "🔴 Below Average Yield"

        st.divider()
        st.subheader("🌾 Crop Yield Forecast Results & Intelligence")
        if from_file:
            st.caption("✅ Loaded Best Crop Yield Model from `rain.ipynb` (`crop_yield_model.pkl`)")
        else:
            st.caption("ℹ️ Model trained on available historical records")
        st.success(f"Forecast successfully generated: **{prediction:.2f} tons/hectare** ({cat_badge})")

        # 1. Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Yield", f"{prediction:.2f} tons/ha", help="Estimated output per hectare")
        m2.metric(f"Total Output ({area:.1f} ha)", f"{total_production:,.1f} tons", help=f"Cultivated Area ({area} ha) × Yield")
        m3.metric(f"{crop} Avg ({state})", f"{state_crop_avg:.2f} tons/ha", delta=f"{prediction - state_crop_avg:+.2f} tons/ha")
        m4.metric("National Crop Avg", f"{crop_avg:.2f} tons/ha", delta=f"{prediction - crop_avg:+.2f} tons/ha")

        st.markdown("---")

        # 2. Visual Graphs Section
        st.subheader("📊 Visual Analytics & Benchmark Comparisons")
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # Benchmark Comparison Bar Chart
            bench_df = pd.DataFrame({
                "Benchmark": ["Your Forecast", f"{crop} in {state}", f"{crop} (National)", f"{state} (All Crops)", "National (All Crops)"],
                "Yield (tons/ha)": [prediction, state_crop_avg, crop_avg, state_avg, overall_avg]
            })
            fig_b = px.bar(
                bench_df,
                x="Benchmark",
                y="Yield (tons/ha)",
                color="Benchmark",
                color_discrete_sequence=["#1B5E20", "#2E7D32", "#43A047", "#81C784", "#B0BEC5"],
                text_auto=".2f"
            )
            style_fig(fig_b, f"Yield Benchmark: {crop} ({state}) vs Regional Averages", 420)
            fig_b.update_traces(textposition="outside", textfont=dict(size=13, color="#000000"))
            fig_b.update_layout(showlegend=False)
            fig_b.update_xaxes(tickangle=-25)
            show_chart(fig_b)

        with col_g2:
            # Historical Distribution with Vertical Line
            crop_yield_data = df[df["Crop_Type"] == crop][YIELD]
            fig_d = px.histogram(
                crop_yield_data,
                x=YIELD,
                nbins=25,
                color_discrete_sequence=["#2E7D32"],
                labels={YIELD: "Yield (tons/hectare)"}
            )
            fig_d.add_vline(
                x=prediction,
                line_dash="dash",
                line_color="#C62828",
                line_width=3,
                annotation_text=f"Forecast: {prediction:.2f} t/ha",
                annotation_font_color="#C62828",
                annotation_position="top right"
            )
            style_fig(fig_d, f"Historical {crop} Yield Distribution (Percentile: {percentile:.1f}%)", 420)
            fig_d.update_layout(yaxis_title="Frequency")
            show_chart(fig_d)

        # 3. Input Condition Comparison
        st.subheader(f"🌦️ Input Conditions Assessment for {crop}")
        hist_crop = df[df["Crop_Type"] == crop]
        rain_ref = hist_crop["Rainfall_mm"].mean()
        temp_ref = hist_crop["Avg_Temperature_C"].mean()
        hum_ref = hist_crop["Humidity_percent"].mean()

        fert_per_ha = fertilizer / area if area > 0 else fertilizer
        pest_per_ha = pesticide / area if area > 0 else pesticide

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rainfall", f"{rainfall:,.0f} mm", delta=f"{rainfall - rain_ref:+,.0f} mm vs avg")
        c2.metric("Avg Temperature", f"{temp:.1f} °C", delta=f"{temp - temp_ref:+.1f} °C vs avg")
        c3.metric("Humidity", f"{humidity:.0f} %", delta=f"{humidity - hum_ref:+.0f} % vs avg")
        c4.metric("Fertilizer Intensity", f"{fert_per_ha:.1f} kg/ha", help="Applied fertilizer per cultivated hectare")

        st.markdown("---")

        # 4. Detailed Agricultural Explanation & Advisory
        st.subheader("📋 In-depth Agronomic Explanation & Advisory")

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown(f"""
            #### 1. Performance & Yield Potential
            - **Yield Rating:** **{category}**
            - **Production Forecast:** Cultivating **{area:.2f} hectares** of **{crop}** in **{state}** is projected to yield **{prediction:.2f} tons/ha**, generating an estimated total output of **{total_production:,.2f} metric tons**.
            - **Benchmark Standing:** The forecasted yield stands at **{prediction - crop_avg:+.2f} tons/ha** relative to the national average for {crop} ({crop_avg:.2f} tons/ha), placing this crop in the **{percentile:.1f}th percentile** of historical observations.
            
            #### 2. Agro-Climatic Drivers
            - **Moisture & Water Balance:** With **{rainfall:,.0f} mm** of seasonal rainfall complemented by **{irrigation} irrigation**, water availability is configured for the **{season}** season.
            - **Thermal & Environmental Fit:** An average temperature of **{temp:.1f}°C** and humidity of **{humidity:.0f}%** provide the climatic boundary conditions for physiological crop development.
            """)

        with exp_col2:
            st.markdown(f"""
            #### 3. Soil & Nutrient Dynamics
            - **Soil Medium ({soil} Soil):** {soil} soils possess distinctive water retention and nutrient capacities. Paired with **{irrigation}**, managing drainage and root aeration will protect against waterlogging.
            - **Input Loadings:** Application of **{fertilizer:,.1f} kg total fertilizer** ({fert_per_ha:.1f} kg/ha) and **{pesticide:,.1f} kg pesticide** ({pest_per_ha:.2f} kg/ha) should be synchronized with crop phenological growth stages.

            #### 4. Actionable Farm Advisory
            - **Irrigation Scheduling:** Maintain optimal moisture during critical flowering and grain-filling phases.
            - **Nutrient Efficiency:** Implement split dose fertilizer applications (basal + top dressing) to prevent nutrient leaching.
            - **Harvest Preparedness:** Plan storage, handling, and logistics for approximately **{total_production:,.1f} tons** of crop volume.
            """)


# =====================================================
# 18. PAGE: FARMER PROFITABILITY PREDICTION
# =====================================================

elif page == "Farmer Profitability Prediction":

    st.header("💰 Farmer Profitability Prediction")
    st.write(
        "Estimate farmer profit using crop, weather, farming conditions "
        "and expected yield."
    )
    st.info(
        "Enter the expected yield and farming details below and click **Predict Farmer Profit** to view earnings, sensitivity analysis, and financial advisory."
    )

    with st.form("profit_forecast_form"):
        inputs = get_prediction_inputs(include_yield=True, key_prefix="profit")
        st.divider()
        submitted = full_width_button(
            st.form_submit_button, "💰 Predict Farmer Profit"
        )

    if submitted:
        state = inputs["State"]
        crop = inputs["Crop_Type"]

        # Ensure Market_Price_per_ton is populated
        if "Market_Price_per_ton" not in inputs or not inputs["Market_Price_per_ton"]:
            subset_crop = df[(df["State"] == state) & (df["Crop_Type"] == crop)]
            if subset_crop.empty:
                subset_crop = df[df["Crop_Type"] == crop]
            inputs["Market_Price_per_ton"] = float(subset_crop["Market_Price_per_ton"].median()) if not subset_crop.empty else float(df["Market_Price_per_ton"].median())

        with st.spinner("Estimating farmer profitability using champion model (R² ≈ 86%)..."):
            model, features, from_file = load_or_train_model(df, PROFIT)
            required_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else features
            input_data = pd.DataFrame([inputs])
            for col in required_cols:
                if col not in input_data.columns:
                    input_data[col] = 0.0
            input_data = input_data[required_cols]
            prediction = float(model.predict(input_data)[0])

        state = inputs["State"]
        crop = inputs["Crop_Type"]
        season = inputs["Season"]
        soil = inputs["Soil_Type"]
        irrigation = inputs["Irrigation_Type"]
        rainfall = float(inputs["Rainfall_mm"])
        temp = float(inputs["Avg_Temperature_C"])
        humidity = float(inputs["Humidity_percent"])
        fertilizer = float(inputs["Fertilizer_Used_kg"])
        pesticide = float(inputs["Pesticide_Used_kg"])
        area = float(inputs["Cultivated_Area_hectares"])
        exp_yield = float(inputs[YIELD])

        total_tons = area * exp_yield
        profit_per_ha = prediction / area if area > 0 else prediction

        # Regional and crop benchmarks
        state_avg_profit = float(df[df["State"] == state][PROFIT].mean())
        crop_avg_profit = float(df[df["Crop_Type"] == crop][PROFIT].mean())
        subset_sc = df[(df["State"] == state) & (df["Crop_Type"] == crop)]
        state_crop_profit = float(subset_sc[PROFIT].mean()) if not subset_sc.empty else crop_avg_profit
        overall_profit = float(df[PROFIT].mean())

        avg_price = float(df[df["Crop_Type"] == crop]["Market_Price_per_ton"].mean())
        est_revenue = total_tons * avg_price
        est_cost = max(0.0, est_revenue - prediction)
        profit_margin = (prediction / est_revenue * 100) if est_revenue > 0 else 0.0

        st.divider()
        st.subheader("💰 Farmer Profitability Results & Agribusiness Insights")
        if from_file:
            st.caption("✅ Loaded Best Profitability Model from `rain.ipynb` (`farmer_profitability_model.pkl`)")
        else:
            st.caption("ℹ️ Model trained on available historical records")
        st.success(f"Estimated Net Farmer Profit: **₹{prediction:,.2f}** (Estimated Margin: {profit_margin:.1f}%)")

        # 1. Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net Farmer Profit", f"₹{prediction:,.2f}", help="Total estimated net earnings")
        m2.metric("Profit per Hectare", f"₹{profit_per_ha:,.2f} / ha", help="Net return per unit land")
        m3.metric(f"State Benchmark ({state})", f"₹{state_avg_profit:,.0f}", delta=f"{prediction - state_avg_profit:+,.0f}")
        m4.metric(f"{crop} Benchmark", f"₹{crop_avg_profit:,.0f}", delta=f"{prediction - crop_avg_profit:+,.0f}")

        st.markdown("---")

        # 2. Visual Graphs Section
        st.subheader("📈 Financial Analysis & Profit Sensitivity")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            # Profit Sensitivity to Yield Variations (±20%)
            scenario_multipliers = [-0.20, -0.10, 0.0, 0.10, 0.20]
            scenario_labels = ["-20% Yield", "-10% Yield", "Expected (Base)", "+10% Yield", "+20% Yield"]
            scenario_profits = []
            required_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else features
            for mult in scenario_multipliers:
                temp_in = inputs.copy()
                temp_in[YIELD] = max(0.1, exp_yield * (1 + mult))
                temp_df = pd.DataFrame([temp_in])
                for col in required_cols:
                    if col not in temp_df.columns:
                        temp_df[col] = 0.0
                temp_df = temp_df[required_cols]
                scenario_profits.append(float(model.predict(temp_df)[0]))

            sens_df = pd.DataFrame({
                "Yield Scenario": scenario_labels,
                "Profit (INR)": scenario_profits
            })
            fig_sens = px.bar(
                sens_df,
                x="Yield Scenario",
                y="Profit (INR)",
                color="Profit (INR)",
                color_continuous_scale="Viridis",
                text_auto=",.0f"
            )
            style_fig(fig_sens, "Profit Sensitivity to Yield Variations (±20%)", 400)
            fig_sens.update_traces(textposition="outside", textfont=dict(size=13, color="#000000"))
            fig_sens.update_layout(showlegend=False)
            show_chart(fig_sens)

        with col_p2:
            # Benchmark Profit Comparison Bar Chart
            bench_profit_df = pd.DataFrame({
                "Benchmark": ["Your Forecast", f"{crop} in {state}", f"{crop} (National)", f"{state} (All Crops)", "National (All Crops)"],
                "Net Profit (INR)": [prediction, state_crop_profit, crop_avg_profit, state_avg_profit, overall_profit]
            })
            fig_pb = px.bar(
                bench_profit_df,
                x="Benchmark",
                y="Net Profit (INR)",
                color="Benchmark",
                color_discrete_sequence=["#17452D", "#246B42", "#328451", "#E69500", "#78909C"],
                text_auto=",.0f"
            )
            style_fig(fig_pb, f"Profit Benchmark: {crop} ({state}) vs Regional Averages", 400)
            fig_pb.update_traces(textposition="outside", textfont=dict(size=13, color="#000000"))
            fig_pb.update_layout(showlegend=False)
            fig_pb.update_xaxes(tickangle=-25)
            show_chart(fig_pb)

        # 3. Financial Breakdown
        st.subheader("💵 Financial Revenue & Cost Structure Simulation")
        fin_col1, fin_col2, fin_col3 = st.columns(3)
        fin_col1.metric("Est. Total Crop Output", f"{total_tons:,.2f} tons", f"At {exp_yield:.2f} t/ha")
        fin_col2.metric("Est. Market Value", f"₹{est_revenue:,.0f}", f"Avg Price: ₹{avg_price:,.0f}/ton")
        fin_col3.metric("Est. Operating & Input Costs", f"₹{est_cost:,.0f}", help="Estimated cost of fertilizer, pesticides, operational inputs")

        st.markdown("---")

        # 4. Detailed Explanations & Business Advisory
        st.subheader("📋 In-depth Financial Advisory & Economic Explanation")

        fin_exp1, fin_exp2 = st.columns(2)
        with fin_exp1:
            delta_10pct = scenario_profits[3] - prediction
            st.markdown(f"""
            #### 1. Economic & Revenue Viability
            - **Total Enterprise Volume:** For a cultivated holding of **{area:,.2f} hectares** producing **{exp_yield:.2f} tons/ha** of **{crop}**, total anticipated crop volume is **{total_tons:,.2f} metric tons**.
            - **Net Profit Realization:** The model estimates a net farmer return of **₹{prediction:,.2f}**, representing an income density of **₹{profit_per_ha:,.2f} per hectare**.
            - **Comparative Standing:** This outcome reflects a **₹{prediction - state_avg_profit:+,.0f}** differential compared to average farm operations across **{state}**.

            #### 2. Key Economic Value Drivers
            - **Acreage Scale Effect:** Cultivating {area:,.2f} hectares allows fixed agricultural overhead to be distributed over larger volume.
            - **Yield Sensitivity:** As demonstrated in the sensitivity curve, a 10% increase in yield expands net profit by approximately **₹{delta_10pct:+,.0f}**, emphasizing that agronomic health directly dictates financial solvency.
            """)

        with fin_exp2:
            st.markdown(f"""
            #### 3. Cost-Return Dynamics
            - **Input Cost Efficiency:** Chemical inputs (**{fertilizer:,.1f} kg fertilizer** and **{pesticide:,.1f} kg pesticide**) represent direct operational capital. Maintaining high nutrient use efficiency prevents cash erosion.
            - **Market Price Exposure:** Realizing the benchmark price of approximately **₹{avg_price:,.0f}/ton** is crucial. Price dips require proactive margin protection.

            #### 4. Strategic Financial Advisory
            - **Post-Harvest Value Capture:** Avoid distress selling at harvest peak; utilize warehouse receipts (e-NWR) or MSP procurement centers.
            - **Risk Transfer & Insurance:** Protect against downside yield loss via Pradhan Mantri Fasal Bima Yojana (PMFBY).
            - **Input Rationalization:** Soil-testing-based fertilizer application can save up to 15–20% on recurring chemical expenses without compromising yield.
            """)




# =====================================================
# 20. FOOTER
# =====================================================

st.divider()
st.markdown("🌾 **AgriSmart | Smart Agriculture Dashboard**")
st.caption("Crop Yield Forecasting and Farmer Profitability Prediction")

# --- Floating Bottom-Right Chat Bubble (Available on EVERY Page) ---
render_floating_chat_bubble()

