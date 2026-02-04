import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Anemia Disease Prediction",
    page_icon="🩸",
    layout="centered"
)

st.title("🩸 Anemia Disease Prediction System")
st.write("AI-based classification of Anemia and its type using blood parameters")

# ======================
# Medical Disclaimer
# ======================
st.warning(
    "⚠️ **Disclaimer:** This is an AI-based screening tool for academic and research purposes only.\n\n"
    "Clinical confirmation is mandatory using further investigations such as:\n"
    "- Iron studies\n"
    "- Peripheral blood smear examination\n"
    "- Ferritin / Fe studies\n"
    "- Hb electrophoresis\n\n"
    "**Do NOT use this tool as a sole diagnostic method.**"
)

# ======================
# Load trained model + label encoder
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("anemia_pipeline.pkl")   # could be Pipeline OR bare estimator
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_model()

# ======================
# Column name mapping (CSV/UI -> training)
# Include common CSV variants too
# ======================
COLUMN_MAPPING = {
    "Age": "Age",
    "Gender": "Gender",

    # RBC variants
    "RBC10¹²-L": "RBC(10^12/L)",
    "RBC(1012/L)": "RBC(10^12/L)",
    "RBC(10^12/L)": "RBC(10^12/L)",

    # HGB variants
    "HGBg-dL": "HGB(g/dl)",
    "HGB(g/dl)": "HGB(g/dl)",

    # HCT variants
    "HCT%": "HCT(%)",
    "HCT(%)": "HCT(%)",

    # MCV variants
    "MCVfL": "MCV(fl)",
    "MCV(fl)": "MCV(fl)",

    # MCH variants
    "MCHpg": "MCH(pg)",
    "MCH(pg)": "MCH(pg)",

    # MCHC variants
    "MCHCg-dL": "MCHC(g/dl)",
    "MCHC(g/dl)": "MCHC(g/dl)",

    # RDW variants
    "RDW-CV%": "RDW-CV(%)",
    "RDW-CV(%)": "RDW-CV(%)",
}

# The exact 9 inputs your RF expects (Male=0, Female=1)
EXPECTED_FEATURES = [
    "Age",
    "Gender",
    "RBC(10^12/L)",
    "HGB(g/dl)",
    "HCT(%)",
    "MCV(fl)",
    "MCH(pg)",
    "MCHC(g/dl)",
    "RDW-CV(%)",
]

# ======================
# Preprocessing
# - renames columns to training format
# - normalizes/encodes Gender (Male=0, Female=1)
# - drops extra columns (e.g., Patient ID)
# - ensures exact feature order (important)
# ======================
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rename known variants
    df = df.rename(columns=COLUMN_MAPPING)

    # Keep only expected features (drop Patient ID or any other extras)
    extra_cols = [c for c in df.columns if c not in EXPECTED_FEATURES]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    # Validate required columns
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize + encode gender
    g = df["Gender"].astype(str).str.strip().str.upper()
    g = g.replace({"M": "MALE", "F": "FEMALE"})
    df["Gender"] = g.map({"MALE": 0, "FEMALE": 1})

    if df["Gender"].isna().any():
        bad = df.loc[df["Gender"].isna(), "Gender"].unique()
        raise ValueError(
            f"Invalid Gender values found: {bad}. Allowed: Male/Female/M/F/MALE/FEMALE"
        )

    # Convert numeric columns
    for col in EXPECTED_FEATURES:
        if col != "Gender":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_cols = [c for c in EXPECTED_FEATURES if df[c].isna().any()]
    if bad_cols:
        raise ValueError(f"Some columns contain invalid/missing values: {bad_cols}")

    # Force exact order
    df = df[EXPECTED_FEATURES]
    return df

# ======================
# Sidebar
# ======================
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input type:", ("Manual Entry", "Upload CSV"))

# ======================
# Manual Input
# ======================
if input_method == "Manual Entry":
    st.subheader("🧪 Enter Patient Details")

    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])

    rbc = st.number_input("RBC (10¹²/L)", value=4.5)
    hgb = st.number_input("HGB (g/dL)", value=13.5)
    hct = st.number_input("HCT (%)", value=40.0)
    mcv = st.number_input("MCV (fL)", value=90.0)
    mch = st.number_input("MCH (pg)", value=30.0)
    mchc = st.number_input("MCHC (g/dL)", value=33.0)
    rdw = st.number_input("RDW-CV (%)", value=13.0)

    if st.button("🔍 Predict"):
        raw_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "RBC10¹²-L": rbc,
            "HGBg-dL": hgb,
            "HCT%": hct,
            "MCVfL": mcv,
            "MCHpg": mch,
            "MCHCg-dL": mchc,
            "RDW-CV%": rdw
        }])

        try:
            X = preprocess_df(raw_df)

            probs = model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))

            disease = le.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx]) * 100

            st.success(f"🧬 Predicted Result: **{disease}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")

            if confidence < 60:
                st.warning("⚠️ Low confidence prediction. Further clinical evaluation recommended.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ======================
# CSV Upload
# ======================
else:
    st.subheader("📂 Upload CSV File")
    st.write("CSV must contain the following columns (extra columns like Patient ID are OK):")
    st.code(["Age", "Gender", "RBC10¹²-L", "HGBg-dL", "HCT%", "MCVfL", "MCHpg", "MCHCg-dL", "RDW-CV%"])

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)

            # Keep a copy for display + downloads (so Patient ID stays visible if present)
            out_df = raw_df.copy()

            # Build features for model
            X = preprocess_df(raw_df)

            preds = model.predict(X)
            probs = model.predict_proba(X)

            out_df["Predicted_Disease"] = le.inverse_transform(preds)
            out_df["Confidence (%)"] = probs.max(axis=1) * 100

            st.success("✅ Prediction completed")
            st.dataframe(out_df)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results",
                data=csv_bytes,
                file_name="anemia_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"CSV processing failed: {e}")

# ======================
# Footer
# ======================
st.markdown("---")
st.caption("🩺 AI-assisted screening tool | Academic & Research Use Only")
