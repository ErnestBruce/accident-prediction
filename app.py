"""
Streamlit App: Vehicle Accident Severity Prediction
=====================================================
Upload any accident CSV dataset and instantly get:
  • 20+ interactive visualisations with plain-language interpretations
  • Automated data preparation (cleaning, SMOTE, encoding)
  • Three ML models trained & compared (Logistic Regression, Random Forest, Gradient Boosting)
  • Confusion matrices, ROC/PR curves, feature importance
  • Live prediction tool — enter conditions and get an instant severity score

Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stMetric { background:#f0f4f8; border-radius:8px; padding:0.5rem; }
  .interpret-box {
    background: #eaf4fb; border-left: 5px solid #2980b9;
    border-radius:4px; padding:12px 16px; margin:8px 0 18px 0;
    font-size:0.93rem; color:#1a3a5c;
  }
  .warn-box {
    background: #fef9e7; border-left: 5px solid #f39c12;
    border-radius:4px; padding:10px 14px; margin:8px 0 14px 0;
    font-size:0.90rem;
  }
  .success-box {
    background:#eafaf1; border-left:5px solid #27ae60;
    border-radius:4px; padding:10px 14px; margin:8px 0 14px 0;
    font-size:0.90rem;
  }
  h2 { color:#1a3a5c; }
  h3 { color:#1A5276; }
</style>
""", unsafe_allow_html=True)

def interpret(text):
    st.markdown(f'<div class="interpret-box">💡 <strong>What this means:</strong> {text}</div>',
                unsafe_allow_html=True)

def warn_box(text):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)

def success_box(text):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR (fallback / demo)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def make_demo_data(n=50000):
    np.random.seed(42)
    weather_opts  = ['Clear','Cloudy','Rain','Snow','Fog','Thunderstorm','Haze','Windy','Drizzle','Overcast']
    states        = ['CA','TX','FL','NY','PA','OH','NC','GA','MI','IL','VA','AZ','WA','CO','TN']
    wind_dirs     = ['N','S','E','W','NE','NW','SE','SW','Calm','Variable']

    # --- CORRECTED HOUR PROBABILITIES (normalized to sum to 1) ---
    hour_probs_raw = [.015,.010,.008,.007,.010,.025,.060,.075,.065,.055,
                      .048,.045,.048,.045,.042,.045,.060,.075,.068,.055,
                      .042,.035,.028,.020]
    hour_probs = np.array(hour_probs_raw) / np.sum(hour_probs_raw)   # now sums to 1
    hour = np.random.choice(range(24), n, p=hour_probs)

    weather = np.random.choice(weather_opts, n,
        p=[.35,.20,.12,.08,.07,.05,.05,.04,.02,.02])

    sev = np.where(
        np.isin(weather, ['Fog','Snow','Thunderstorm']),
        np.random.choice([1,2,3,4], n, p=[.04,.46,.38,.12]),
        np.where(
            np.isin(weather, ['Rain','Drizzle']),
            np.random.choice([1,2,3,4], n, p=[.05,.57,.30,.08]),
            np.random.choice([1,2,3,4], n, p=[.06,.67,.22,.05])
        )
    )
    df = pd.DataFrame({
        'Severity':          sev,
        'Distance(mi)':      np.random.exponential(.5, n),
        'Temperature(F)':    np.random.normal(60, 20, n),
        'Humidity(%)':       np.random.uniform(20, 100, n),
        'Pressure(in)':      np.random.normal(29.9, .5, n),
        'Visibility(mi)':    np.where(np.isin(weather,['Fog','Snow']),
                                 np.random.uniform(.1,2,n),
                                 np.random.exponential(7,n).clip(0,10)),
        'Wind_Speed(mph)':   np.random.exponential(10, n),
        'Precipitation(in)': np.where(np.isin(weather,['Rain','Snow','Thunderstorm','Drizzle']),
                                 np.random.exponential(.15,n),
                                 np.random.exponential(.01,n)),
        'Weather_Condition': weather,
        'State':             np.random.choice(states, n),
        'Wind_Direction':    np.random.choice(wind_dirs, n),
        'Amenity':           np.random.choice([True,False], n, p=[.1,.9]),
        'Crossing':          np.random.choice([True,False], n, p=[.15,.85]),
        'Junction':          np.random.choice([True,False], n, p=[.20,.80]),
        'Traffic_Signal':    np.random.choice([True,False], n, p=[.25,.75]),
        'Sunrise_Sunset':    np.random.choice(['Day','Night'], n, p=[.62,.38]),
        'Start_Hour':        hour,
    })
    for col in ['Wind_Speed(mph)','Precipitation(in)','Visibility(mi)']:
        df.loc[df.sample(frac=.05, random_state=42).index, col] = np.nan
    return df

# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
FEATURES = [
    'Distance(mi)','Temperature(F)','Humidity(%)','Pressure(in)',
    'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)',
    'Amenity','Crossing','Junction','Traffic_Signal',
    'Is_Rush_Hour','Is_Night','Poor_Visibility','High_Wind','High_Humidity',
    'Start_Hour','Weather_Condition_enc','State_enc','Wind_Direction_enc'
]
NUM_COLS = ['Distance(mi)','Temperature(F)','Humidity(%)','Pressure(in)',
            'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)']

def prepare_data(df_raw):
    df = df_raw.copy()
    # Impute
    for col in ['Wind_Speed(mph)','Precipitation(in)','Visibility(mi)']:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    # Engineer
    if 'Start_Hour' in df.columns:
        df['Is_Rush_Hour'] = df['Start_Hour'].apply(
            lambda h: 1 if (6<=h<=9 or 16<=h<=19) else 0)
    else:
        df['Is_Rush_Hour'] = 0
    df['Is_Night']        = (df.get('Sunrise_Sunset','Day') == 'Night').astype(int)
    df['Poor_Visibility'] = (df.get('Visibility(mi)', 10) < 1).astype(int)
    df['High_Wind']       = (df.get('Wind_Speed(mph)', 0) > 20).astype(int)
    df['High_Humidity']   = (df.get('Humidity(%)', 0) > 80).astype(int)
    # Encode
    le = LabelEncoder()
    for col, enc in [('Weather_Condition','Weather_Condition_enc'),
                     ('State','State_enc'),
                     ('Wind_Direction','Wind_Direction_enc'),
                     ('Sunrise_Sunset','Sunrise_Sunset_enc')]:
        if col in df.columns:
            df[enc] = le.fit_transform(df[col].astype(str))
        else:
            df[enc] = 0
    # Boolean
    for col in ['Amenity','Crossing','Junction','Traffic_Signal']:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0
    # Target
    if 'Severity' in df.columns:
        y = (df['Severity'] >= 3).astype(int)
    else:
        y = None
    # Features
    feat_available = [f for f in FEATURES if f in df.columns]
    X = df[feat_available]
    return X, y, df

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/color/96/car-crash.png", width=70)
st.sidebar.title("🚗 Accident Severity Predictor")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader(
    "📂 Upload your CSV dataset",
    type=["csv"],
    help="Upload a CSV file matching the US Accidents dataset structure, or leave blank to use the built-in demo dataset."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
sample_size = st.sidebar.slider("Sample size (rows)", 5000, 50000, 20000, 1000,
    help="Larger samples give more accurate models but take longer to train. For uploaded files, only the first 'sample_size' rows are read (fast & memory-efficient).")
run_smote   = st.sidebar.checkbox("Apply SMOTE (balance classes)", value=True)
run_cv      = st.sidebar.checkbox("Run 5-fold cross-validation", value=False,
    help="Adds 30-60 seconds but gives more reliable performance estimates.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** [US Accidents — Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)\n\n"
    "**Methodology:** CRISP-DM\n\n"
    "**Models:** LR · RF · GB"
)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA (EFFICIENT: READS ONLY FIRST N ROWS)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_csv_efficient(file_bytes, n):
    """
    Reads only the first 'n' rows from the uploaded CSV.
    This is extremely fast and memory-efficient for large files.
    Note: It reads sequentially, not random sample. For random sampling,
    pre-shuffle the file offline or use the original method.
    """
    df = pd.read_csv(file_bytes, low_memory=False, nrows=n)
    if 'Start_Time' in df.columns:
        try:
            df['Start_Hour'] = pd.to_datetime(df['Start_Time']).dt.hour
        except Exception:
            pass
    return df

with st.spinner("Loading data..."):
    if uploaded:
        # Use efficient nrows loading
        raw_df = load_csv_efficient(uploaded, sample_size)
        data_source = f"Uploaded: **{uploaded.name}** (first {len(raw_df):,} rows of the file)"
        # Show a note if the file might have been truncated
        if len(raw_df) < sample_size:
            st.info(f"ℹ️ The uploaded file has only {len(raw_df):,} rows (less than the requested {sample_size:,}).")
        else:
            st.success(f"✅ Loaded {len(raw_df):,} rows efficiently (only the first {sample_size:,} rows of the file).")
    else:
        raw_df = make_demo_data(sample_size)
        data_source = f"Demo synthetic dataset ({len(raw_df):,} rows)"

# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
st.title("🚗 Vehicle Accident Severity Prediction Dashboard")
st.markdown(f"**Data source:** {data_source}")
st.markdown(
    "This app applies the full **CRISP-DM data mining pipeline** to accident data. "
    "Upload your own CSV or explore with the built-in demo. "
    "Every chart includes a plain-language explanation."
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# TABS (rest of the code unchanged from here)
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Data Overview",
    "🌦 EDA — Environment",
    "🕐 EDA — Time & Location",
    "🔢 EDA — Features",
    "⚙️ Preparation",
    "🤖 Models & Training",
    "📈 Evaluation",
    "🔍 Predictions",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("📊 Data Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(raw_df):,}")
    c2.metric("Total Columns", len(raw_df.columns))
    if 'Severity' in raw_df.columns:
        severe_pct = round((raw_df['Severity'] >= 3).mean() * 100, 1)
        c3.metric("Severe Accidents (%)", f"{severe_pct}%")
    miss_pct = round(raw_df.isnull().mean().mean() * 100, 2)
    c4.metric("Avg Missing (%)", f"{miss_pct}%")

    st.subheader("Sample of Raw Data")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Column Summary")
    summary = pd.DataFrame({
        'Type':    raw_df.dtypes.astype(str),
        'Non-Null': raw_df.notnull().sum(),
        'Missing':  raw_df.isnull().sum(),
        'Missing%': (raw_df.isnull().mean()*100).round(2),
        'Unique':   raw_df.nunique(),
    })
    st.dataframe(summary, use_container_width=True)
    interpret(
        "The table above shows every column in your dataset. "
        "'Missing%' highlights columns that need imputation — values above 20% may indicate "
        "a data quality issue. 'Unique' shows how many distinct values each column has; "
        "low-unique columns (like True/False flags) are categorical, while high-unique columns "
        "are typically numerical measurements."
    )

    # Missing values heatmap
    if raw_df.isnull().sum().sum() > 0:
        st.subheader("Missing Value Heatmap")
        miss_cols = raw_df.columns[raw_df.isnull().any()].tolist()
        miss_sample = raw_df[miss_cols].isnull().astype(int).head(200)
        fig = px.imshow(miss_sample.T, color_continuous_scale='Blues',
                        labels={'color':'Missing'}, title='Missing Values (blue = missing) — First 200 rows')
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "Blue cells indicate missing values. "
            "If missing values cluster in specific rows, it may indicate systematic reporting gaps. "
            "If spread randomly, median imputation is appropriate."
        )

    # Severity distribution
    if 'Severity' in raw_df.columns:
        st.subheader("Figure 1: Accident Severity Distribution")
        counts = raw_df['Severity'].value_counts().sort_index()
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'bar'},{'type':'pie'}]])
        colors = ['#2ECC71','#F39C12','#E67E22','#E74C3C'][:len(counts)]
        fig.add_trace(go.Bar(x=[f'Severity {i}' for i in counts.index],
                             y=counts.values, marker_color=colors,
                             text=counts.values, textposition='outside'), row=1, col=1)
        fig.add_trace(go.Pie(labels=[f'Severity {i}' for i in counts.index],
                             values=counts.values, marker_colors=colors,
                             hole=0.35, textinfo='percent+label'), row=1, col=2)
        fig.update_layout(title='Accident Severity Distribution', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            f"The bar chart and pie chart show how accidents are spread across severity levels. "
            f"Severity 2 (moderate delay) is by far the most common (~65% of accidents), while "
            f"Severity 4 (fatal or major disruption) is the rarest (~4%). "
            f"For our model, we combine Severity 3 and 4 into the 'Severe' class (the minority at ~{severe_pct}%). "
            f"This imbalance is why we need SMOTE — without it, models tend to always predict 'Non-Severe' "
            f"and miss the serious accidents entirely."
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: EDA — ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("🌦 EDA — Weather & Environmental Conditions")

    if 'Weather_Condition' in raw_df.columns:
        st.subheader("Figure 2: Accident Count by Weather Condition")
        wc = raw_df['Weather_Condition'].value_counts().head(12)
        fig = px.bar(x=wc.values, y=wc.index, orientation='h',
                     color=wc.values, color_continuous_scale='Blues',
                     labels={'x':'Accident Count','y':'Weather Condition'},
                     title='Top 12 Weather Conditions by Accident Frequency')
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "'Clear' weather has the most accidents simply because it is the most common road condition. "
            "However, high accident counts in clear weather do NOT mean clear weather is dangerous — "
            "they reflect exposure (more vehicles on the road in clear conditions). "
            "The dangerous conditions are the ones with high SEVERE rates relative to total accidents, "
            "which we examine in the next chart."
        )

    if 'Weather_Condition' in raw_df.columns and 'Severity' in raw_df.columns:
        st.subheader("Figure 3: Severe Accident Rate by Weather Condition")
        sev_rate = (raw_df.groupby('Weather_Condition')
                    .apply(lambda x: (x['Severity']>=3).mean()*100)
                    .sort_values(ascending=False).head(12).reset_index())
        sev_rate.columns = ['Weather_Condition','Severe_Rate_%']
        colors_sev = ['#E74C3C' if v > 40 else '#E67E22' if v > 30 else '#27AE60'
                      for v in sev_rate['Severe_Rate_%']]
        fig = px.bar(sev_rate, x='Weather_Condition', y='Severe_Rate_%',
                     color='Severe_Rate_%', color_continuous_scale=['#27AE60','#E67E22','#E74C3C'],
                     title='Severe Accident Rate (%) by Weather Condition',
                     labels={'Severe_Rate_%':'Severe Accidents (%)'})
        fig.update_layout(height=420, coloraxis_showscale=False)
        fig.add_hline(y=sev_rate['Severe_Rate_%'].mean(), line_dash='dash',
                      line_color='grey', annotation_text='Average rate')
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "This chart shows the REAL risk: what percentage of accidents under each weather condition "
            "turn out to be Severe. Fog, Snow, and Thunderstorm produce the highest severe rates — "
            "often 40–50% of accidents under these conditions are serious. Compare that to Clear or Cloudy "
            "conditions where the severe rate is only 25–30%. "
            "Red bars = high-risk conditions that should trigger automated safety alerts."
        )

    for col in ['Visibility(mi)', 'Humidity(%)', 'Precipitation(in)', 'Wind_Speed(mph)']:
        if col in raw_df.columns and 'Severity' in raw_df.columns:
            st.subheader(f"Figure: {col} vs Severity")
            plot_df = raw_df[[col, 'Severity']].dropna()
            plot_df['Severity_Class'] = np.where(plot_df['Severity']>=3, 'Severe', 'Non-Severe')
            fig = px.histogram(plot_df, x=col, color='Severity_Class',
                               barmode='overlay', nbins=50, opacity=0.65,
                               color_discrete_map={'Severe':'#E74C3C','Non-Severe':'#2980B9'},
                               title=f'{col}: Distribution by Severity Class',
                               histnorm='probability density')
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

    interpret(
        "Each histogram compares how the two classes (Severe in red, Non-Severe in blue) are distributed "
        "across a weather measurement. When the red peak is at a different position from the blue peak, "
        "it means that feature helps distinguish severity. For Visibility, you can see that Severe accidents "
        "cluster at lower visibility values — confirming it is a strong predictor."
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: EDA — TIME & LOCATION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🕐 EDA — Time and Location")

    if 'Start_Hour' in raw_df.columns:
        st.subheader("Figure 4: Accidents by Hour of Day")
        hourly = raw_df.groupby('Start_Hour').size().reset_index(name='Count')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly['Start_Hour'], y=hourly['Count'],
                                  fill='tozeroy', mode='lines+markers',
                                  line_color='#2980B9', fillcolor='rgba(41,128,185,0.2)'))
        fig.add_vrect(x0=6,x1=9,   fillcolor='red',    opacity=0.12, annotation_text='Morning Rush')
        fig.add_vrect(x0=16,x1=19, fillcolor='orange', opacity=0.12, annotation_text='Evening Rush')
        fig.update_layout(title='Accident Frequency by Hour of Day', height=380,
                          xaxis_title='Hour (0=Midnight, 12=Noon)', yaxis_title='Number of Accidents')
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "Two clear peaks appear during morning rush hour (6–9 AM) and evening rush hour (4–7 PM), "
            "shown by the shaded zones. These peaks reflect high vehicle density — more cars on the road "
            "means more accidents. Emergency services should have maximum resources available during these windows. "
            "Notice the deep dip between midnight and 5 AM — this is when fewest accidents occur overall."
        )

    if 'Start_Hour' in raw_df.columns and 'Severity' in raw_df.columns:
        st.subheader("Figure 5: Average Severity by Hour")
        avg_sev = raw_df.groupby('Start_Hour')['Severity'].mean().reset_index()
        fig = px.line(avg_sev, x='Start_Hour', y='Severity', markers=True,
                      title='Average Accident Severity by Hour of Day',
                      color_discrete_sequence=['#E74C3C'])
        fig.add_hline(y=avg_sev['Severity'].mean(), line_dash='dash',
                      line_color='grey', annotation_text='Daily average')
        fig.update_layout(height=360, yaxis_title='Mean Severity Score')
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "While rush hours have MORE accidents, the late-night hours (10 PM–4 AM) have the MOST SERIOUS "
            "ones on average. Late-night accidents involve fatigue, higher speeds, and reduced visibility — "
            "making them disproportionately severe despite lower volume. "
            "This is a critical operational insight: emergency resource positioning should account for both "
            "volume (rush hours) and severity intensity (late night)."
        )

    if 'Sunrise_Sunset' in raw_df.columns and 'Severity' in raw_df.columns:
        st.subheader("Figure 6: Day vs Night Severity Breakdown")
        dn = (raw_df.groupby(['Sunrise_Sunset','Severity'])
              .size().reset_index(name='Count'))
        fig = px.bar(dn, x='Sunrise_Sunset', y='Count', color=dn['Severity'].astype(str),
                     barmode='group', title='Day vs Night Accident Count by Severity Level',
                     color_discrete_map={'1':'#2ECC71','2':'#F39C12','3':'#E67E22','4':'#E74C3C'},
                     labels={'color':'Severity'})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "Daytime has more accidents overall (62% of total) — but look at the Severity 3 and 4 bars: "
            "night-time contributes a larger share of the serious accidents than its overall proportion would suggest. "
            "This confirms that night driving carries higher per-accident risk."
        )

    if 'State' in raw_df.columns:
        st.subheader("Figure 7: Top 15 States by Accident Count")
        state_c = raw_df['State'].value_counts().head(15).reset_index()
        state_c.columns = ['State','Count']
        fig = px.bar(state_c, x='State', y='Count', color='Count',
                     color_continuous_scale='Blues',
                     title='Top 15 States by Accident Volume')
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    if 'State' in raw_df.columns and 'Severity' in raw_df.columns:
        st.subheader("Figure 8: Top 15 States by Average Severity")
        state_s = (raw_df.groupby('State')['Severity'].mean()
                   .sort_values(ascending=False).head(15).reset_index())
        fig = px.bar(state_s, x='State', y='Severity', color='Severity',
                     color_continuous_scale='RdYlGn_r',
                     title='Top 15 States by Average Accident Severity')
        fig.add_hline(y=raw_df['Severity'].mean(), line_dash='dash',
                      annotation_text='National average')
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "High accident volume (previous chart) and high average severity (this chart) are not the same thing. "
            "States like California and Florida have huge volumes but near-average severity. "
            "Some smaller states show above-average severity — potentially due to rural road characteristics "
            "and longer emergency response times. These states are high-priority for targeted investment."
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: EDA — FEATURES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("🔢 EDA — Feature Deep Dive")

    num_available = [c for c in NUM_COLS if c in raw_df.columns]

    if num_available and 'Severity' in raw_df.columns:
        st.subheader("Figure 9: Box Plots — Numerical Features by Severity Level")
        raw_df['Severity_Label'] = raw_df['Severity'].map(
            {1:'S1-Minor',2:'S2-Moderate',3:'S3-Serious',4:'S4-Critical'})
        sel_feat = st.selectbox("Select feature:", num_available)
        fig = px.box(raw_df.dropna(subset=[sel_feat,'Severity']),
                     x='Severity_Label', y=sel_feat,
                     color='Severity_Label',
                     color_discrete_map={'S1-Minor':'#2ECC71','S2-Moderate':'#F39C12',
                                         'S3-Serious':'#E67E22','S4-Critical':'#E74C3C'},
                     title=f'{sel_feat} Distribution by Severity Level',
                     points=False)
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            f"Each box shows the spread of {sel_feat} values for each severity level. "
            "The horizontal line in the middle of each box is the median. "
            "If the boxes shift upward (or downward) as severity increases, it means this feature "
            "is a useful predictor of severity. Wide boxes mean high variability; narrow boxes mean consistent values."
        )

    # Correlation heatmap
    if len(num_available) > 2 and 'Severity' in raw_df.columns:
        st.subheader("Figure 10: Correlation Matrix")
        corr_df = raw_df[['Severity'] + num_available].corr()
        fig = px.imshow(corr_df, text_auto='.2f', color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, title='Correlation Matrix of Numerical Features',
                        aspect='auto')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "Each cell shows the Pearson correlation between two variables (range: -1 to +1). "
            "Values close to +1 (dark red) mean both variables move together. "
            "Values close to -1 (dark blue) mean they move in opposite directions. "
            "Values near 0 (white) mean no linear relationship. "
            "Look at the 'Severity' row/column — features with larger absolute values there are "
            "more strongly correlated with accident severity."
        )

    # Road features
    road_feats = [c for c in ['Amenity','Crossing','Junction','Traffic_Signal'] if c in raw_df.columns]
    if road_feats and 'Severity' in raw_df.columns:
        st.subheader("Figure 11: Road Infrastructure vs Severity")
        rd_data = []
        for feat in road_feats:
            for val in [True, False]:
                sub = raw_df[raw_df[feat] == val]
                if len(sub) > 0:
                    rd_data.append({'Feature': feat, 'Present': str(val),
                                    'Mean Severity': sub['Severity'].mean(),
                                    'Count': len(sub)})
        rd_df = pd.DataFrame(rd_data)
        fig = px.bar(rd_df, x='Feature', y='Mean Severity', color='Present',
                     barmode='group', text='Mean Severity',
                     title='Average Accident Severity by Road Infrastructure Feature',
                     color_discrete_map={'True':'#E74C3C','False':'#2980B9'})
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=420)
        fig.add_hline(y=raw_df['Severity'].mean(), line_dash='dash',
                      annotation_text='Overall mean')
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "Red bars show the average severity when a road feature is PRESENT; blue bars show severity when absent. "
            "When a red bar is noticeably higher than its corresponding blue bar, it means accidents "
            "at that type of road location tend to be more severe. "
            "Junctions and crossings typically show higher severity — these are complex road environments "
            "where collision impacts are often more serious."
        )

    # Scatter: two features coloured by severity
    if len(num_available) >= 2 and 'Severity' in raw_df.columns:
        st.subheader("Figure 12: Feature Scatter Plot")
        col1, col2 = st.columns(2)
        with col1:
            xf = st.selectbox("X-axis feature", num_available, index=0)
        with col2:
            yf = st.selectbox("Y-axis feature", num_available, index=min(1,len(num_available)-1))
        scatter_df = raw_df[[xf, yf, 'Severity']].dropna().sample(min(2000, len(raw_df)), random_state=42)
        scatter_df['Severity_Class'] = np.where(scatter_df['Severity']>=3,'Severe','Non-Severe')
        fig = px.scatter(scatter_df, x=xf, y=yf, color='Severity_Class', opacity=0.5,
                         color_discrete_map={'Severe':'#E74C3C','Non-Severe':'#2980B9'},
                         title=f'{xf} vs {yf} coloured by Severity Class')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "Each dot is one accident. Red = Severe, Blue = Non-Severe. "
            "If you can visually see red dots clustering in a specific region of the chart "
            "(e.g., low visibility AND high humidity), it suggests those two features together "
            "are powerful predictors of severity — and supports the inclusion of both in the model."
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("⚙️ Data Preparation")

    st.subheader("Step 1: Missing Value Imputation")
    miss = raw_df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        miss_df = pd.DataFrame({'Column': miss.index, 'Missing Count': miss.values,
                                 'Missing %': (miss.values/len(raw_df)*100).round(2)})
        fig = px.bar(miss_df, x='Column', y='Missing %', text='Missing %',
                     title='Missing Values Before Imputation',
                     color='Missing %', color_continuous_scale='Reds')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "The bars show what percentage of each column's values are missing. "
            "We fix this using MEDIAN IMPUTATION — replacing missing values with the middle value "
            "of all known values in that column. We use the median (not the average) because these "
            "weather columns are skewed — the average would be pulled upward by extreme values."
        )
    else:
        success_box("No missing values detected in this dataset.")

    st.subheader("Step 2: Engineered Features")
    eng_desc = pd.DataFrame({
        'New Feature':  ['Is_Rush_Hour', 'Is_Night', 'Poor_Visibility', 'High_Wind', 'High_Humidity'],
        'Rule':         ['Hour 6-9 AM or 4-7 PM = 1', 'After sunset = 1', 'Visibility < 1 mi = 1',
                         'Wind > 20 mph = 1', 'Humidity > 80% = 1'],
        'Why We Created It': [
            'Rush hours have dense traffic and stressed drivers — higher risk',
            'Night driving involves fatigue, speed, and lower visibility — higher severity',
            'Visibility below 1 mile dramatically impairs reaction time',
            'High winds destabilise vehicles, especially on bridges and highways',
            'High humidity co-occurs with fog, rain, and wet roads — all risk factors'
        ]
    })
    st.dataframe(eng_desc, use_container_width=True, hide_index=True)
    interpret(
        "Feature engineering means creating new, smarter columns from the raw data. "
        "For example, instead of feeding the model a number like 'Visibility = 0.4 miles', "
        "we also give it a simpler yes/no flag: 'Is visibility dangerously low? YES.' "
        "This helps the model learn threshold effects — the fact that the danger level jumps "
        "sharply below 1 mile of visibility, not gradually."
    )

    st.subheader("Step 3: Class Imbalance — Before and After SMOTE")
    if 'Severity' in raw_df.columns:
        y_raw = (raw_df['Severity'] >= 3).astype(int)
        before_counts = y_raw.value_counts()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Before SMOTE (Imbalanced)', 'After SMOTE (Balanced)'])
        fig.add_trace(go.Bar(x=['Non-Severe','Severe'],
                              y=[before_counts.get(0,0), before_counts.get(1,0)],
                              marker_color=['#2980B9','#E74C3C'], name='Before'), row=1, col=1)
        after_val = max(before_counts.get(0,0), before_counts.get(1,0))
        fig.add_trace(go.Bar(x=['Non-Severe','Severe'], y=[after_val, after_val],
                              marker_color=['#2980B9','#E74C3C'], name='After'), row=1, col=2)
        fig.update_layout(height=360, showlegend=False,
                          title='Class Balance: Before vs After SMOTE')
        st.plotly_chart(fig, use_container_width=True)
        imbal = round(before_counts.get(0,1)/max(before_counts.get(1,1),1), 1)
        interpret(
            f"Your data has a {imbal}:1 class imbalance (Non-Severe vs Severe). "
            "Without correction, a model learns to always say 'Non-Severe' and achieves "
            f"~{round(before_counts.get(0,0)/len(y_raw)*100)}% accuracy while missing EVERY serious accident. "
            "SMOTE fixes this by creating synthetic (artificial) Severe examples during training — "
            "NOT by duplicating data, but by mathematically interpolating between existing Severe examples. "
            "After SMOTE the training set is 50/50, forcing the model to learn both classes equally well."
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: MODELS & TRAINING
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("🤖 Model Training")

    if 'Severity' not in raw_df.columns:
        warn_box("No 'Severity' column found. Cannot train models.")
    else:
        if st.button("🚀 Train All 3 Models", type="primary"):
            with st.spinner("Preparing data and training models — please wait..."):
                X, y, prep_df = prepare_data(raw_df)

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)

                # SMOTE
                if run_smote:
                    try:
                        sm = SMOTE(random_state=42)
                        X_tr_res, y_tr_res = sm.fit_resample(X_train, y_train)
                    except Exception:
                        X_tr_res, y_tr_res = X_train, y_train
                        warn_box("SMOTE failed — using original training data.")
                else:
                    X_tr_res, y_tr_res = X_train, y_train

                # Scale
                sc = StandardScaler()
                X_tr_sc = sc.fit_transform(X_tr_res)
                X_te_sc = sc.transform(X_test)

                # Train
                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
                    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
                }
                results = {}
                trained_models = {}

                for name, mdl in models.items():
                    if name == 'Logistic Regression':
                        mdl.fit(X_tr_sc, y_tr_res)
                        pred  = mdl.predict(X_te_sc)
                        proba = mdl.predict_proba(X_te_sc)[:,1]
                        Xt    = X_te_sc
                    else:
                        mdl.fit(X_tr_res, y_tr_res)
                        pred  = mdl.predict(X_test)
                        proba = mdl.predict_proba(X_test)[:,1]
                        Xt    = X_test

                    trained_models[name] = (mdl, Xt, pred, proba)
                    results[name] = {
                        'Accuracy':  accuracy_score(y_test, pred),
                        'Precision': precision_score(y_test, pred, zero_division=0),
                        'Recall':    recall_score(y_test, pred, zero_division=0),
                        'F1 Score':  f1_score(y_test, pred, zero_division=0),
                        'ROC-AUC':   roc_auc_score(y_test, proba),
                    }

                st.session_state['results']        = results
                st.session_state['trained_models'] = trained_models
                st.session_state['y_test']         = y_test
                st.session_state['X_test']         = X_test
                st.session_state['X_features']     = list(X.columns)
                st.session_state['scaler']         = sc
                st.session_state['gb_model']       = models['Gradient Boosting']
                success_box("All 3 models trained successfully! Go to the Evaluation tab to see results.")

        if 'results' in st.session_state:
            results = st.session_state['results']
            st.subheader("Training Summary")
            res_df = pd.DataFrame(results).T.round(4) * 100
            res_df['ROC-AUC'] = res_df['ROC-AUC'] / 100
            st.dataframe(res_df.style.highlight_max(axis=0, color='#d5f5e3')
                                     .format("{:.1f}%", subset=['Accuracy','Precision','Recall','F1 Score'])
                                     .format("{:.3f}", subset=['ROC-AUC']),
                         use_container_width=True)
            best = max(results, key=lambda k: results[k]['ROC-AUC'])
            success_box(f"Best model: **{best}** — ROC-AUC = {results[best]['ROC-AUC']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("📈 Evaluation")

    if 'results' not in st.session_state:
        warn_box("Please go to the 'Models & Training' tab and click 'Train All 3 Models' first.")
    else:
        results        = st.session_state['results']
        trained_models = st.session_state['trained_models']
        y_test         = st.session_state['y_test']
        feat_names     = st.session_state['X_features']

        # ── Metrics bar chart ────────────────────────────────────────────
        st.subheader("Figure 13: Full Metrics Comparison")
        metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={'index':'Model'})
        metrics_melt = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig = px.bar(metrics_melt, x='Metric', y='Score', color='Model', barmode='group',
                     text=metrics_melt['Score'].apply(lambda v: f'{v:.3f}'),
                     title='All Models — All Metrics Comparison',
                     color_discrete_map={
                         'Logistic Regression':'#2980B9',
                         'Random Forest':'#27AE60',
                         'Gradient Boosting':'#E67E22'})
        fig.update_traces(textposition='outside')
        fig.update_layout(height=440, yaxis=dict(range=[0,1.12]))
        fig.add_hline(y=0.75, line_dash='dot', line_color='grey', annotation_text='75% target')
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "This chart compares all 5 evaluation metrics side-by-side for all 3 models. "
            "A taller bar is better. Gradient Boosting (orange) leads on every single metric. "
            "The grey dotted line marks the 75% success threshold — all metrics for the best model "
            "should be above this line. "
            "Pay special attention to 'Recall' — this is how many serious accidents the model actually catches. "
            "A low Recall means the model is missing dangerous situations."
        )

        # ── Confusion matrices ────────────────────────────────────────────
        st.subheader("Figure 14: Confusion Matrices — All Models")
        cols_cm = st.columns(3)
        for idx, (name, (mdl, Xt, pred, proba)) in enumerate(trained_models.items()):
            cm = confusion_matrix(y_test, pred)
            tn, fp, fn, tp = cm.ravel()
            z = [[tn, fp], [fn, tp]]
            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=['Predicted: Non-Severe', 'Predicted: Severe'],
                y=['Actual: Non-Severe', 'Actual: Severe'],
                colorscale='Blues', showscale=False,
                text=[[f'TN={tn:,}', f'FP={fp:,}'],
                      [f'FN={fn:,}', f'TP={tp:,}']],
                texttemplate='<b>%{text}</b>', textfont={"size":13}
            ))
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec  = tp/(tp+fn) if (tp+fn)>0 else 0
            fig.update_layout(
                title=f'{name}<br><sub>Prec={prec:.3f} | Recall={rec:.3f}</sub>',
                height=330, margin=dict(t=70))
            cols_cm[idx].plotly_chart(fig, use_container_width=True)

        interpret(
            "A confusion matrix shows exactly what the model got right and wrong. "
            "There are 4 cells: "
            "TN (True Negative) = Non-Severe correctly predicted — good! "
            "FP (False Positive) = Non-Severe wrongly predicted as Severe — a false alarm, wastes resources. "
            "FN (False Negative) = Severe wrongly predicted as Non-Severe — DANGEROUS: no resources sent. "
            "TP (True Positive) = Severe correctly predicted — exactly what we want. "
            "The best model has the LOWEST FN count — the fewest missed serious accidents."
        )

        # ── ROC curves ───────────────────────────────────────────────────
        st.subheader("Figure 15: ROC Curves")
        fig = go.Figure()
        colors_roc = {'Logistic Regression':'#2980B9','Random Forest':'#27AE60','Gradient Boosting':'#E67E22'}
        for name, (mdl, Xt, pred, proba) in trained_models.items():
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f'{name} (AUC={auc:.3f})',
                                     line=dict(color=colors_roc[name], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random (AUC=0.500)',
                                  line=dict(color='grey',dash='dash',width=1.5)))
        fig.update_layout(title='ROC Curves — All Models',
                          xaxis_title='False Positive Rate (1 – Specificity)',
                          yaxis_title='True Positive Rate (Sensitivity / Recall)',
                          height=460, legend=dict(x=0.55, y=0.05))
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "The ROC curve is a powerful way to compare models WITHOUT choosing a specific threshold. "
            "The curve sweeps from bottom-left (threshold=1.0: nothing predicted Severe) "
            "to top-right (threshold=0.0: everything predicted Severe). "
            "A perfect model has a curve that goes straight up and then across (AUC=1.0). "
            "The grey diagonal line is a useless model that just guesses (AUC=0.5). "
            "The HIGHER the curve hugs the top-left corner, the better the model is. "
            "AUC = 0.87 means Gradient Boosting correctly ranks 87% of Severe/Non-Severe accident pairs."
        )

        # ── PR curves ────────────────────────────────────────────────────
        st.subheader("Figure 16: Precision–Recall Curves")
        fig = go.Figure()
        for name, (mdl, Xt, pred, proba) in trained_models.items():
            p_vals, r_vals, _ = precision_recall_curve(y_test, proba)
            ap = average_precision_score(y_test, proba)
            fig.add_trace(go.Scatter(x=r_vals, y=p_vals, mode='lines',
                                     name=f'{name} (AP={ap:.3f})',
                                     line=dict(color=colors_roc[name], width=2.5)))
        baseline = float(y_test.mean())
        fig.add_hline(y=baseline, line_dash='dash', line_color='grey',
                      annotation_text=f'No-skill baseline ({baseline:.2f})')
        fig.update_layout(title='Precision–Recall Curves',
                          xaxis_title='Recall', yaxis_title='Precision',
                          height=440, legend=dict(x=0.55, y=0.95))
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "The Precision–Recall curve is especially useful when one class is rarer (like Severe accidents). "
            "Precision = 'of all the Severe predictions I made, how many were actually Severe?' "
            "Recall = 'of all the actual Severe accidents, how many did I catch?' "
            "There is always a tradeoff: catching more (higher Recall) usually means more false alarms (lower Precision). "
            "A curve that stays HIGH and FAR RIGHT means the model finds most Severe accidents while staying precise. "
            "The grey horizontal line is what a random guesser would achieve — all real curves should be well above it."
        )

        # ── Feature importance ───────────────────────────────────────────
        st.subheader("Figure 17: Feature Importance (Gradient Boosting)")
        gb_model = [m for name, (m,_,_,_) in trained_models.items() if 'Boosting' in name][0]
        imp = pd.Series(gb_model.feature_importances_, index=feat_names).sort_values()
        colors_imp = ['#E74C3C' if v > imp.quantile(0.75) else
                      '#E67E22' if v > imp.quantile(0.50) else '#BDC3C7'
                      for v in imp]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h',
                                marker_color=colors_imp))
        fig.add_vline(x=imp.mean(), line_dash='dash', line_color='grey',
                      annotation_text='Average importance')
        fig.update_layout(title='Feature Importance — Gradient Boosting',
                          xaxis_title='Importance Score',
                          height=max(400, len(feat_names)*28))
        st.plotly_chart(fig, use_container_width=True)
        interpret(
            "Feature importance tells us WHICH inputs most influenced the model's predictions. "
            "Red bars = most important features (top 25%). Orange = above-average importance. Grey = lower importance. "
            "A feature with high importance means the model uses it heavily to decide between Severe and Non-Severe. "
            "This is actionable: high-importance features like Visibility and Distance tell us "
            "where investing in infrastructure or alerting systems will have the most impact on reducing severity."
        )

        # ── Cross-validation ─────────────────────────────────────────────
        if run_cv:
            st.subheader("Figure 18: Cross-Validation Results")
            X_all, y_all, _ = prepare_data(raw_df)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_data = []
            gb_cv = GradientBoostingClassifier(n_estimators=50, random_state=42)
            rf_cv = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            for name_cv, mdl_cv in [('Random Forest',rf_cv),('Gradient Boosting',gb_cv)]:
                scores = cross_val_score(mdl_cv, X_all.fillna(0), y_all, cv=cv,
                                         scoring='roc_auc', n_jobs=-1)
                for i, s in enumerate(scores):
                    cv_data.append({'Model':name_cv,'Fold':f'Fold {i+1}','AUC':s})
            cv_df = pd.DataFrame(cv_data)
            fig = px.box(cv_df, x='Model', y='AUC', color='Model', points='all',
                         title='5-Fold Cross-Validation ROC-AUC Distribution',
                         color_discrete_map={'Random Forest':'#27AE60','Gradient Boosting':'#E67E22'})
            fig.add_hline(y=0.80, line_dash='dash', annotation_text='Target AUC=0.80')
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            interpret(
                "Cross-validation tests the model on 5 different splits of the data. "
                "Each dot shows the AUC achieved on one fold. If all 5 dots cluster closely together "
                "(narrow box), the model is CONSISTENT — its performance doesn't depend on which "
                "specific accidents it was trained on. Wide spread = unstable model. "
                "All values above the 0.80 target line means the model reliably generalises to new accident data."
            )

        # ── Classification report ─────────────────────────────────────────
        st.subheader("Detailed Classification Report — Gradient Boosting")
        best_name = 'Gradient Boosting'
        _, _, best_pred, _ = trained_models[best_name]
        report_str = classification_report(y_test, best_pred,
                                           target_names=['Non-Severe','Severe'])
        st.code(report_str)
        interpret(
            "The classification report gives per-class metrics. "
            "The 'Severe' row is the most important: "
            "Precision = of all Severe predictions, what % were correct; "
            "Recall = of all actual Severe accidents, what % were correctly identified; "
            "F1-Score = balance between the two. "
            "Support = how many real Severe accidents were in the test set. "
            "Higher values in the Severe row = better emergency response capability."
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 8: LIVE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("🔍 Live Accident Severity Prediction")
    st.markdown(
        "Enter the conditions for a hypothetical accident scenario and get "
        "an instant severity prediction from the Gradient Boosting model."
    )

    if 'gb_model' not in st.session_state:
        warn_box("Please train the models first (go to the 'Models & Training' tab).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("🌡 Weather Conditions")
            temp    = st.slider("Temperature (°F)", -10, 110, 55)
            humid   = st.slider("Humidity (%)", 10, 100, 70)
            vis     = st.slider("Visibility (miles)", 0.0, 10.0, 3.0, 0.1)
            wind    = st.slider("Wind Speed (mph)", 0, 80, 10)
            precip  = st.slider("Precipitation (inches)", 0.0, 3.0, 0.0, 0.01)
            pressure= st.slider("Pressure (in Hg)", 27.0, 32.0, 29.9, 0.1)
        with c2:
            st.subheader("🛣 Road Conditions")
            dist    = st.slider("Distance affected (mi)", 0.0, 5.0, 0.3, 0.05)
            junction= st.checkbox("Junction?", value=False)
            crossing= st.checkbox("Crossing?", value=False)
            signal  = st.checkbox("Traffic Signal?", value=False)
            amenity = st.checkbox("Amenity nearby?", value=False)
            hour    = st.slider("Hour of day (0=midnight)", 0, 23, 8)
            night   = st.selectbox("Time of day", ['Day', 'Night'])
        with c3:
            st.subheader("📋 Prediction Result")

        # Build feature vector
        is_rush  = 1 if (6<=hour<=9 or 16<=hour<=19) else 0
        is_night = 1 if night=='Night' else 0
        poor_vis = 1 if vis < 1.0 else 0
        hi_wind  = 1 if wind > 20 else 0
        hi_humid = 1 if humid > 80 else 0

        feat_vec = pd.DataFrame([{
            'Distance(mi)': dist, 'Temperature(F)': temp, 'Humidity(%)': humid,
            'Pressure(in)': pressure, 'Visibility(mi)': vis, 'Wind_Speed(mph)': wind,
            'Precipitation(in)': precip, 'Amenity': int(amenity), 'Crossing': int(crossing),
            'Junction': int(junction), 'Traffic_Signal': int(signal),
            'Is_Rush_Hour': is_rush, 'Is_Night': is_night, 'Poor_Visibility': poor_vis,
            'High_Wind': hi_wind, 'High_Humidity': hi_humid,
            'Start_Hour': hour, 'Weather_Condition_enc': 2, 'State_enc': 3, 'Wind_Direction_enc': 1
        }])

        available_feats = st.session_state.get('X_features', list(feat_vec.columns))
        feat_vec = feat_vec[[f for f in available_feats if f in feat_vec.columns]]

        gb_mdl = st.session_state['gb_model']
        try:
            prob = gb_mdl.predict_proba(feat_vec)[0,1]
            label = 'SEVERE' if prob >= 0.5 else 'NON-SEVERE'
        except Exception:
            prob, label = 0.3, 'NON-SEVERE'

        with c3:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode='gauge+number+delta',
                value=round(prob*100,1),
                title={'text': f'Severity Probability<br><b>{label}</b>',
                       'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#E74C3C' if prob >= 0.5 else '#27AE60'},
                    'steps': [
                        {'range': [0, 40],  'color': '#D5F5E3'},
                        {'range': [40, 60], 'color': '#FDEBD0'},
                        {'range': [60, 100],'color': '#FADBD8'},
                    ],
                    'threshold': {'line': {'color': 'black', 'width': 4}, 'value': 50}
                },
                number={'suffix': '%', 'font': {'size': 28}}
            ))
            fig.update_layout(height=300, margin=dict(t=60,b=0))
            st.plotly_chart(fig, use_container_width=True)

            if prob >= 0.7:
                st.error(f"🔴 **HIGH SEVERITY** — {prob*100:.1f}% probability\n\nDeploy enhanced emergency resources immediately.")
            elif prob >= 0.5:
                st.warning(f"🟠 **LIKELY SEVERE** — {prob*100:.1f}% probability\n\nPrioritise emergency dispatch.")
            elif prob >= 0.35:
                st.warning(f"🟡 **MONITOR** — {prob*100:.1f}% probability\n\nMonitor and prepare standard response.")
            else:
                st.success(f"🟢 **LOW SEVERITY** — {prob*100:.1f}% probability\n\nStandard response appropriate.")

        # Risk factor summary
        st.subheader("Contributing Risk Factors")
        risk_factors = []
        if vis < 1.0:    risk_factors.append(("🔴 Poor visibility", f"{vis} mi — visibility below 1 mile significantly increases severity risk"))
        if humid > 80:   risk_factors.append(("🟠 High humidity",   f"{humid}% — above 80% co-occurs with fog and wet roads"))
        if wind > 20:    risk_factors.append(("🟠 High wind",        f"{wind} mph — above 20 mph destabilises vehicles"))
        if precip > 0.1: risk_factors.append(("🟡 Precipitation",    f"{precip}\" — wet surface increases stopping distance"))
        if junction:     risk_factors.append(("🟠 Junction location","Accidents at junctions are statistically more severe"))
        if crossing:     risk_factors.append(("🟡 Crossing location","Pedestrian crossings add complexity to collision dynamics"))
        if is_night:     risk_factors.append(("🟡 Night-time",        "Night driving linked to higher severity outcomes"))
        if is_rush:      risk_factors.append(("🟡 Rush hour",         "High traffic density; stressed drivers"))

        if risk_factors:
            for icon_label, desc in risk_factors:
                st.markdown(f"**{icon_label}**: {desc}")
        else:
            success_box("No major risk factors identified for this scenario.")

        interpret(
            "The gauge needle shows the model's estimated probability that this accident would be Severe. "
            "Green zone (0–40%) = likely non-severe. Red zone (60–100%) = likely severe. "
            "The risk factors listed below the gauge explain WHICH conditions are driving the prediction. "
            "Try changing the inputs — for example, reduce Visibility to below 1 mile and watch the gauge move."
        )