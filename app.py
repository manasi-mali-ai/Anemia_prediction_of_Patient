import streamlit as st
import pandas as pd
import numpy as np
import joblib

# data preprocessing and label encoding
def preprocess_df(df: pd.DataFrame, model):
    df = df.rename(columns=COLUMN_MAPPING).copy()

    if "Gender" in df.columns:
        g = df["Gender"].astype(str).str.strip().str.upper()
        g = g.replace({"M": "MALE", "F": "FEMALE"})
        df["Gender_norm"] = g.map({"MALE": "Male", "FEMALE": "Female"})

        if df["Gender_norm"].isna().any():
            bad = df.loc[df["Gender_norm"].isna(), "Gender"].unique()
            raise ValueError(f"Invalid Gender values found: {bad}. Allowed: Male/Female/M/F/MALE/FEMALE")

        if not hasattr(model, "named_steps"):
            # ✅ Your training encoding:
            # Male = 0, Female = 1
            df["Gender"] = df["Gender_norm"].map({"Male": 0, "Female": 1}).astype(int)
        else:
            df["Gender"] = df["Gender_norm"]

        df = df.drop(columns=["Gender_norm"])

    for col in df.columns:
        if col != "Gender":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_cols = [c for c in df.columns if c != "Gender" and df[c].isna().any()]
    if bad_cols:
        raise ValueError(f"Some numeric columns contain invalid/missing values: {bad_cols}")

    return df


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
# Load trained pipeline
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("anemia_pipeline.pkl")   # full sklearn Pipeline
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_model()

# ======================
# Column name mapping
# (UI / CSV ➜ training format)
# ======================
COLUMN_MAPPING = {
    "Age": "Age",
    "Gender": "Gender",
    "RBC10¹²-L": "RBC(10^12/L)",
    "HGBg-dL": "HGB(g/dl)",
    "HCT%": "HCT(%)",
    "MCVfL": "MCV(fl)",
    "MCHpg": "MCH(pg)",
    "MCHCg-dL": "MCHC(g/dl)",
    "RDW-CV%": "RDW-CV(%)"
}

# ======================
# Sidebar
# ======================
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose input type:",
    ("Manual Entry", "Upload CSV")
)

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
        input_df = pd.DataFrame([{
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

        # Rename to training column names
        input_df = input_df.rename(columns=COLUMN_MAPPING)
        input_df = preprocess_df(input_df, model)


        # Predict (PIPELINE handles everything)
        probs = model.predict_proba(input_df)[0]
        pred_idx = np.argmax(probs)

        disease = le.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100

        st.success(f"🧬 Predicted Result: **{disease}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

        if confidence < 60:
            st.warning("⚠️ Low confidence prediction. Further clinical evaluation recommended.")

# ======================
# CSV Upload
# ======================
else:
    st.subheader("📂 Upload CSV File")
    st.write("CSV must contain the following columns:")
    st.code(list(COLUMN_MAPPING.keys()))

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Rename columns
        df = df.rename(columns=COLUMN_MAPPING)
        df = preprocess_df(df, model)


        # Predict
        preds = model.predict(df)
        probs = model.predict_proba(df)

        df["Predicted_Disease"] = le.inverse_transform(preds)
        df["Confidence (%)"] = probs.max(axis=1) * 100

        st.success("✅ Prediction completed")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results",
            data=csv,
            file_name="anemia_predictions.csv",
            mime="text/csv"
        )

# ======================
# Footer
# ======================
st.markdown("---")
st.caption("🩺 AI-assisted screening tool | Academic & Research Use Only")
