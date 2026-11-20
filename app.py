import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pytorch_tabnet.tab_model import TabNetRegressor

# ------------------------------------------------------------
# Student Performance Predictor - dashboard built from the
# uploaded notebook's feature and target definitions.
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.environ.get("STUDENT_DATA_PATH", "stud_synthetic_50000_full.csv")
MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "student_tabnet_bundle.joblib")
TARGETS = ["math_score", "reading_score", "writing_score"]

CAT_COLS = [
    "gender", "race_ethnicity", "parental_level_of_education", "lunch",
    "test_preparation_course", "school", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic",
]
NUM_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2", "G3",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

# Colours / styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #f6f7fb; color: #20223a; }
/* Force readable dark text on the light dashboard background. */
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] { color: #20223a !important; }
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4,
[data-testid="stMain"] p, [data-testid="stMain"] li,
[data-testid="stMain"] label, [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
  color: #20223a !important;
}
[data-testid="stMain"] [data-testid="stAlert"] p,
[data-testid="stMain"] [data-testid="stAlert"] div { color: #24324a !important; }
[data-testid="stSidebar"] { background: #ffffff; }
[data-testid="stSidebar"] * { color: #252447; }
/* Keep the hero banner's intentional white typography. */
.hero, .hero h1, .hero p, .hero div { color: #ffffff !important; }
.hero p { color: #e9e7ff !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
.hero {
  background: linear-gradient(110deg,#3026a8 0%,#6250df 58%,#8a55e8 100%);
  padding: 26px 30px; border-radius: 20px; color: white; margin-bottom: 20px;
  box-shadow: 0 12px 30px rgba(69,57,180,.18);
}
.hero h1 { color:white; font-size:2.0rem; margin:0 0 6px 0; font-weight:800; }
.hero p { color:#e9e7ff; margin:0; font-size:1rem; }
.metric-card {
  background:white; border:1px solid #ececf5; border-radius:15px;
  padding:16px 18px; min-height:105px; box-shadow:0 3px 12px rgba(30,30,70,.035);
}
.metric-label { color:#73758a; font-size:.82rem; margin-bottom:8px; }
.metric-value { color:#191b35; font-size:1.65rem; font-weight:800; }
.metric-note { color:#6a58d5; font-size:.78rem; margin-top:4px; }
.panel {
  background:white; border:1px solid #ececf5; border-radius:16px;
  padding:18px 20px; box-shadow:0 3px 12px rgba(30,30,70,.035);
}
.section-title { font-size:1.05rem; font-weight:750; color:#252447; margin-bottom:8px; }
.small-muted { color:#77798d; font-size:.83rem; }
div[data-testid="stMetric"] {
  background:white; border:1px solid #ececf5; padding:14px 16px;
  border-radius:14px; box-shadow:0 3px 12px rgba(30,30,70,.035);
}
.stButton>button[kind="primary"] {
  background:linear-gradient(90deg,#5145cd,#7650df); border:0; border-radius:10px;
  font-weight:700; min-height:44px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(path):
    data = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + TARGETS if c not in data.columns]
    if missing:
        raise ValueError("Dataset is missing required notebook columns: " + ", ".join(missing))
    return data

def grade_band(avg):
    if avg >= 90: return "A", "Excellent", "Low"
    if avg >= 80: return "B", "Good", "Low"
    if avg >= 70: return "C", "Fair", "Moderate"
    if avg >= 60: return "D", "Needs improvement", "High"
    return "F", "At risk", "High"

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first",
                                         sparse_output=False)),
            ]), CAT_COLS),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]), NUM_COLS),
        ],
        remainder="drop",
    )

def train_tabnet(data, progress=None):
    clean = data.dropna(subset=TARGETS).copy()
    for c in CAT_COLS:
        clean[c] = clean[c].fillna("missing").astype(str)

    X = clean[FEATURE_COLS].copy()
    bundle = {"model_name": "TabNet", "targets": {}, "feature_cols": FEATURE_COLS}
    metric_rows = []

    for idx, target in enumerate(TARGETS):
        y = clean[target].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        prep = make_preprocessor()
        X_train_t = prep.fit_transform(X_train).astype(np.float32)
        X_test_t = prep.transform(X_test).astype(np.float32)
        y_train_t = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
        y_test_t = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

        model = TabNetRegressor(
            n_d=16, n_a=16, n_steps=5, gamma=1.3, lambda_sparse=1e-3,
            optimizer_params={"lr": 0.02}, seed=42, verbose=0
        )
        model.fit(
            X_train=X_train_t, y_train=y_train_t,
            eval_set=[(X_test_t, y_test_t)],
            eval_name=["test"], eval_metric=["mae"],
            max_epochs=100, patience=15, batch_size=1024,
            virtual_batch_size=128, num_workers=0, drop_last=False,
        )
        pred = model.predict(X_test_t).reshape(-1)
        r2 = float(r2_score(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        bundle["targets"][target] = {
            "preprocessor": prep, "model": model,
            "metrics": {"R2": r2, "MAE": mae, "RMSE": rmse},
        }
        metric_rows.append({"Subject": target.replace("_score","").title(),
                            "R²": r2, "MAE": mae, "RMSE": rmse})
        if progress:
            progress.progress((idx + 1) / len(TARGETS),
                              text=f"Training TabNet for {target.replace('_',' ')} ({idx+1}/3)")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle, pd.DataFrame(metric_rows)

def predict_one(bundle, profile):
    row = pd.DataFrame([{c: profile[c] for c in FEATURE_COLS}])
    scores = {}
    for target in TARGETS:
        item = bundle["targets"][target]
        x = item["preprocessor"].transform(row).astype(np.float32)
        pred = float(item["model"].predict(x).reshape(-1)[0])
        scores[target] = float(np.clip(pred, 0, 100))
    avg = float(np.mean(list(scores.values())))
    letter, label, risk = grade_band(avg)
    return scores, avg, letter, label, risk

def recommendations(profile, risk):
    tips = []
    if int(profile["failures"]) > 0:
        tips.append("Review earlier failed subjects and arrange targeted academic support.")
    if int(profile["studytime"]) <= 2:
        tips.append("Build a regular study routine and increase focused study time gradually.")
    if int(profile["absences"]) > 10:
        tips.append("Review the reasons for missed classes and make an attendance support plan.")
    if profile["test_preparation_course"] == "none":
        tips.append("Consider structured exam preparation and regular practice tests.")
    if profile["schoolsup"] == "no" and int(profile["failures"]) > 0:
        tips.append("Ask the school about additional learning support or tutoring.")
    if profile["famsup"] == "no":
        tips.append("Explore appropriate family, mentor, or peer support for study planning.")
    if not tips:
        tips.append("Maintain consistent study habits, attendance, and regular progress reviews.")
    if risk == "High":
        tips.insert(0, "Arrange an educator review before making any high-impact decision.")
    return tips

# Load dataset
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.info("Place the CSV beside app.py, or set STUDENT_DATA_PATH to its full path.")
    st.stop()

# Header
st.markdown("""
<div class="hero">
  <div style="font-size:.78rem;font-weight:700;letter-spacing:.08em;opacity:.9">MACHINE LEARNING • ACADEMIC ANALYTICS</div>
  <h1>🎓 Student Performance Predictor</h1>
  <p>Predict subject scores, explore student patterns, and identify learners who may need early academic support.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("## 🎓 Student Analytics")
page = st.sidebar.radio("Navigate", ["Dashboard Overview", "Predict Student", "Model Performance", "Dataset Explorer"])
st.sidebar.markdown("---")
st.sidebar.caption("Model: TabNet regression | Targets: Math, Reading, Writing")
st.sidebar.caption("Risk bands are derived from the mean predicted score; they are not a separate risk-classification model.")

# Model load status
bundle = None
if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        st.sidebar.warning(f"Saved model could not be loaded: {e}")

if page == "Dashboard Overview":
    total = len(df)
    mean_score = df[TARGETS].mean().mean()
    overall = df[TARGETS].mean(axis=1)
    high_share = float((overall < 60).mean() * 100)
    ready = bundle is not None

    c1,c2,c3,c4 = st.columns(4)
    for col, label, value, note in [
        (c1,"STUDENT RECORDS",f"{total:,}","Uploaded dataset"),
        (c2,"AVERAGE SUBJECT SCORE",f"{mean_score:.1f}/100","Across three exam scores"),
        (c3,"BELOW 60 SCORE BAND",f"{high_share:.1f}%","Dataset descriptive statistic"),
        (c4,"PREDICTION MODEL","Ready" if ready else "Not trained","TabNet • 3 subject models"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

    st.write("")
    left,right = st.columns([1.05,1])
    with left:
        st.markdown('<div class="panel"><div class="section-title">Subject score distribution</div>',unsafe_allow_html=True)
        long_df = df[TARGETS].melt(var_name="Subject",value_name="Score")
        long_df["Subject"] = long_df["Subject"].str.replace("_score","",regex=False).str.title()
        fig = px.histogram(long_df, x="Score", color="Subject", barmode="overlay",
                           nbins=25, opacity=.72, color_discrete_sequence=["#6250df","#25a89a","#f1ad45"])
        fig.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="white",plot_bgcolor="white",legend_title="")
        st.plotly_chart(fig,width="stretch")
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="section-title">Average score by study time</div>',unsafe_allow_html=True)
        study = df.groupby("studytime")[TARGETS].mean().reset_index().melt("studytime",var_name="Subject",value_name="Average score")
        study["Subject"] = study["Subject"].str.replace("_score","",regex=False).str.title()
        fig2 = px.line(study,x="studytime",y="Average score",color="Subject",markers=True,
                       color_discrete_sequence=["#6250df","#25a89a","#f1ad45"])
        fig2.update_layout(height=330,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="white",plot_bgcolor="white",
                           xaxis_title="Study time (1–4 scale)",legend_title="")
        st.plotly_chart(fig2,width="stretch")
        st.markdown('</div>',unsafe_allow_html=True)

    st.write("")
    a,b = st.columns([1.2,.8])
    with a:
        st.markdown('<div class="panel"><div class="section-title">Performance by test preparation</div>',unsafe_allow_html=True)
        prep = df.groupby("test_preparation_course")[TARGETS].mean().reset_index()
        prep = prep.melt("test_preparation_course",var_name="Subject",value_name="Average score")
        prep["Subject"] = prep["Subject"].str.replace("_score","",regex=False).str.title()
        fig3=px.bar(prep,x="test_preparation_course",y="Average score",color="Subject",barmode="group",
                    color_discrete_sequence=["#6250df","#25a89a","#f1ad45"])
        fig3.update_layout(height=300,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="white",plot_bgcolor="white",legend_title="")
        st.plotly_chart(fig3,width="stretch")
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="section-title">Project workflow</div>',unsafe_allow_html=True)
        st.markdown("""
        **01** Dataset collection · 50,000 records  
        ↓  
        **02** Missing-value handling and encoding  
        ↓  
        **03** Numerical feature scaling  
        ↓  
        **04** TabNet model training (3 targets)  
        ↓  
        **05** R², MAE and RMSE evaluation  
        ↓  
        **06** Interactive subject-score prediction
        """)
        st.caption("The dashboard uses the feature and target definitions in your uploaded notebook.")
        st.markdown('</div>',unsafe_allow_html=True)

elif page == "Predict Student":
    st.subheader("🧑‍🎓 Predict an individual student's performance")
    st.caption("Enter the student's profile. Inputs follow the columns used by your notebook.")
    if bundle is None:
        st.warning("The TabNet models have not been trained yet. Go to **Model Performance** and train the models first.")
    # options directly from the supplied dataset
    defaults = {
        "gender":"female","race_ethnicity":"group C","parental_level_of_education":"some college",
        "lunch":"standard","test_preparation_course":"none","school":"GP","address":"U",
        "famsize":"GT3","Pstatus":"T","Mjob":"other","Fjob":"other","reason":"course",
        "guardian":"mother","schoolsup":"no","famsup":"yes","paid":"no","activities":"no",
        "nursery":"yes","higher":"yes","internet":"yes","romantic":"no",
        "age":17,"Medu":3,"Fedu":2,"traveltime":1,"studytime":2,"failures":0,
        "famrel":4,"freetime":3,"goout":3,"Dalc":1,"Walc":2,"health":4,
        "absences":4,"G1":11,"G2":11,"G3":11,
    }
    with st.form("student_form"):
        st.markdown("#### Student and family details")
        c1,c2,c3 = st.columns(3)
        profile={}
        with c1:
            for col in ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course","school","address"]:
                opts=sorted(df[col].dropna().astype(str).unique().tolist())
                val=defaults.get(col,opts[0])
                profile[col]=st.selectbox(col.replace("_"," ").title(),opts,index=opts.index(val) if val in opts else 0)
        with c2:
            for col in ["famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup"]:
                opts=sorted(df[col].dropna().astype(str).unique().tolist())
                val=defaults.get(col,opts[0])
                profile[col]=st.selectbox(col.replace("_"," ").title(),opts,index=opts.index(val) if val in opts else 0)
        with c3:
            for col in ["paid","activities","nursery","higher","internet","romantic"]:
                opts=sorted(df[col].dropna().astype(str).unique().tolist())
                val=defaults.get(col,opts[0])
                profile[col]=st.selectbox(col.replace("_"," ").title(),opts,index=opts.index(val) if val in opts else 0)

        st.markdown("#### Academic, lifestyle and family factors")
        ncols=st.columns(4)
        numeric_labels = {
            "age":"Age","Medu":"Mother's education (0–4)","Fedu":"Father's education (0–4)",
            "traveltime":"Travel time (1–4)","studytime":"Study time (1–4)",
            "failures":"Past failures","famrel":"Family relationship (1–5)",
            "freetime":"Free time (1–5)","goout":"Going out (1–5)",
            "Dalc":"Workday alcohol (1–5)","Walc":"Weekend alcohol (1–5)",
            "health":"Health (1–5)","absences":"Absences","G1":"Prior grade G1 (0–20)",
            "G2":"Prior grade G2 (0–20)","G3":"Prior grade G3 (0–20)",
        }
        for i,col in enumerate(NUM_COLS):
            with ncols[i%4]:
                lo=int(df[col].min()); hi=int(df[col].max())
                d=int(np.clip(defaults[col],lo,hi))
                profile[col]=st.number_input(numeric_labels[col],min_value=lo,max_value=hi,value=d,step=1)
        submitted=st.form_submit_button("✨ Predict Student Performance",type="primary",width="stretch")

    if submitted:
        if bundle is None:
            st.error("Train the TabNet models first, then return to this page.")
        else:
            scores,avg,letter,label,risk=predict_one(bundle,profile)
            st.write("")
            rcol1,rcol2,rcol3,rcol4=st.columns(4)
            for col,title,val in [
                (rcol1,"MATH SCORE",f"{scores['math_score']:.1f}"),
                (rcol2,"READING SCORE",f"{scores['reading_score']:.1f}"),
                (rcol3,"WRITING SCORE",f"{scores['writing_score']:.1f}"),
                (rcol4,"OVERALL AVERAGE",f"{avg:.1f}"),
            ]:
                col.metric(title,val)
            color="#168b62" if risk=="Low" else "#d18a18" if risk=="Moderate" else "#c44755"
            st.markdown(f"""
            <div style="background:white;border:1px solid #ececf5;border-radius:16px;padding:22px;margin-top:12px">
              <div style="color:#77798d;font-size:.85rem">PREDICTED PERFORMANCE BAND</div>
              <div style="font-size:2rem;font-weight:800;color:{color}">{letter} — {label}</div>
              <div style="color:{color};font-weight:700">Academic support indicator: {risk}</div>
              <div style="color:#77798d;font-size:.82rem;margin-top:8px">This band is calculated from the average of the three predicted subject scores. It is not a calibrated probability of failure.</div>
            </div>
            """,unsafe_allow_html=True)
            chart=pd.DataFrame({"Subject":["Math","Reading","Writing"],"Predicted score":[scores[t] for t in TARGETS]})
            fig=px.bar(chart,x="Subject",y="Predicted score",text_auto=".1f",range_y=[0,100],
                       color="Subject",color_discrete_sequence=["#6250df","#25a89a","#f1ad45"])
            fig.update_layout(showlegend=False,height=300,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="white",plot_bgcolor="white")
            st.plotly_chart(fig,width="stretch")
            st.markdown("#### Suggested next steps")
            for tip in recommendations(profile,risk):
                st.markdown(f"- {tip}")

elif page == "Model Performance":
    st.subheader("🧠 TabNet model training and evaluation")
    st.write("This page trains one TabNet regression model for each target defined in your notebook: Math, Reading and Writing.")
    st.info("Training uses an 80/20 train-test split (random_state=42), one-hot encoding for categorical features, imputation and standard scaling for numeric features. The dashboard reports test-set R², MAE and RMSE.")
    if bundle is not None:
        rows=[]
        for target,item in bundle["targets"].items():
            m=item["metrics"]
            rows.append({"Subject":target.replace("_score","").title(),"R²":m["R2"],"MAE":m["MAE"],"RMSE":m["RMSE"]})
        metrics_df=pd.DataFrame(rows)
        st.success("Saved TabNet model bundle found.")
        c1,c2,c3=st.columns(3)
        c1.metric("Mean R²",f"{metrics_df['R²'].mean():.3f}")
        c2.metric("Mean MAE",f"{metrics_df['MAE'].mean():.2f}")
        c3.metric("Mean RMSE",f"{metrics_df['RMSE'].mean():.2f}")
        st.dataframe(metrics_df.style.format({"R²":"{:.4f}","MAE":"{:.3f}","RMSE":"{:.3f}"}),width="stretch",hide_index=True)
        fig=px.bar(metrics_df.melt("Subject",var_name="Metric",value_name="Value"),x="Subject",y="Value",color="Metric",barmode="group",
                   color_discrete_sequence=["#6250df","#25a89a","#f1ad45"])
        fig.update_layout(height=330,margin=dict(l=10,r=10,t=15,b=10),paper_bgcolor="white",plot_bgcolor="white")
        st.plotly_chart(fig,width="stretch")
    else:
        st.warning("No trained model file found yet.")
    if st.button("🚀 Train / Retrain TabNet Models",type="primary",width="stretch"):
        bar=st.progress(0,text="Preparing data...")
        try:
            trained,metrics_df=train_tabnet(df,bar)
            st.success("Training completed. The three subject models and their evaluation metrics have been saved.")
            st.dataframe(metrics_df.style.format({"R²":"{:.4f}","MAE":"{:.3f}","RMSE":"{:.3f}"}),width="stretch",hide_index=True)
            st.info("The model bundle is saved at artifacts/student_tabnet_bundle.joblib. Refresh the page or reopen the prediction tab to use it.")
            st.rerun()
        except Exception as e:
            st.exception(e)
            st.error("Training did not complete. Check the error above and confirm the dependencies are installed.")

elif page == "Dataset Explorer":
    st.subheader("📊 Dataset Explorer")
    st.caption("Exploratory summaries are calculated directly from your uploaded CSV.")
    c1,c2,c3=st.columns(3)
    c1.metric("Rows",f"{len(df):,}")
    c2.metric("Columns",len(df.columns))
    c3.metric("Missing cells",f"{int(df.isna().sum().sum()):,}")
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(100),width="stretch")
    st.markdown("#### Missing values")
    miss=df.isna().sum().reset_index()
    miss.columns=["Column","Missing values"]
    miss=miss[miss["Missing values"]>0].sort_values("Missing values",ascending=False)
    if miss.empty:
        st.success("No missing values found in the uploaded dataset.")
    else:
        st.dataframe(miss,width="stretch",hide_index=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(),width="stretch")

st.markdown("---")
st.caption("Student Performance Predictor • Built using the uploaded 50,000-row dataset and notebook feature definitions. Predictions are estimates for educational support, not decisions about students.")
