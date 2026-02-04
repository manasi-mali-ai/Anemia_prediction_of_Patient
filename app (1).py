import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD TRAINED OBJECTS
# ===============================
model = joblib.load("/content/anemia_model.pkl")
scaler = joblib.load("/content/scaler.pkl")
imputer = joblib.load("/content/imputer.pkl")
encoder = joblib.load("/content/label_encoder.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Anemia Classification System",
    layout="centered"
)

# ===============================
# TITLE
# ===============================
st.title("🩸 Anemia Classification System")

# ===============================
# MEDICAL DISCLAIMER
# ===============================
st.warning("""
⚠️ **Medical Disclaimer**

This is an **AI-based tool to classify anemia**.

The prediction is intended **only for screening and academic purposes**.
Final confirmation **must be done by healthcare professionals** using
further clinical investigations such as:

• Iron studies
• Peripheral blood smear examination
• Ferritin / Fe studies
• Hb electrophoresis
• Other relevant laboratory tests

❗ **This tool should NOT be used as a substitute for medical diagnosis.**
""")

st.markdown("---")

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["📂 Upload CSV", "✍️ Manual Patient Entry"])

# ===============================
# TAB 1: CSV UPLOAD
# ===============================
with tab1:
    st.subheader("Upload Patient CSV File")

    uploaded_file = st.file_uploader(
        "Upload CSV file containing patient data",
        type=["csv"]
    )

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Encode Gender
        data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})

        # Drop PatientID if exists
        X = data.drop(columns=['PatientID'], errors='ignore')

        # Handle missing values + scaling
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        # Prediction
        predictions = model.predict(X_imputed)
        data['Predicted_Anemia_Type'] = encoder.inverse_transform(predictions)

        st.success("✅ Prediction completed successfully")
        st.dataframe(data)

# ===============================
# TAB 2: MANUAL ENTRY
# ===============================
with tab2:
    st.subheader("Enter Patient Details Manually")

    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["MALE", "FEMALE"])
    rbc = st.number_input("RBC (10¹²/L)")
    hgb = st.number_input("HGB (g/dL)")
    hct = st.number_input("HCT (%)")
    mcv = st.number_input("MCV (fL)")
    mch = st.number_input("MCH (pg)")
    mchc = st.number_input("MCHC (g/dL)")
    rdw = st.number_input("RDW-CV (%)")

    if st.button("🔍 Predict Anemia"):
        gender_encoded = 0 if gender == "MALE" else 1

        input_df = pd.DataFrame([[
            age, gender_encoded, rbc, hgb, hct, mcv, mch, mchc, rdw
        ]], columns=[
            'Age', 'Gender', 'RBC', 'HGB', 'HCT',
            'MCV', 'MCH', 'MCHC', 'RDW'
        ])

        # Impute + scale
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Predict
        prediction = encoder.inverse_transform(
            model.predict(input_imputed)
        )[0]

        if prediction == "No Anemia":
            st.success("✅ **Patient is NOT Anemic**")
        else:
            st.error(f"⚠️ **Predicted Anemia Type:** {prediction}")
