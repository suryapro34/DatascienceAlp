import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import os

# Page Configuration
st.set_page_config(page_title="CO2 Emission Predictor", layout="wide")
st.title("🚗 Vehicle CO2 Emission Predictor (ICE Vehicles)")

# Load Dataset
@st.cache_data
def load_data():
    if not os.path.exists('EV_vs_ICE_Vehicle_Specs_2015_2026.csv'):
        st.error("⚠️ Dataset file not found. Please make sure `EV_vs_ICE_Vehicle_Specs_2015_2026.csv` is in the same folder.")
        return None
    df = pd.read_csv('EV_vs_ICE_Vehicle_Specs_2015_2026.csv')
    df['Fuel_Type'] = df['Fuel_Type'].fillna('Unknown')
    df = df.drop_duplicates()
    return df

df = load_data()
if df is None:
    st.stop()

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.y_test = None
    st.session_state.y_pred = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

# Sidebar: Model Settings
st.sidebar.header("⚙️ Settings")
test_size = st.sidebar.slider("Test Data Size", 0.1, 0.4, 0.25, 0.05)
model_type = st.sidebar.radio("Select Model", ["Simple Linear Regression (SLR)", "Multiple Linear Regression (MLR)"])
train_btn = st.sidebar.button("🚀 Train Model")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Load the dataset → Adjust parameters → Train the model → View predictions in the Prediction tab.")

# Main Layout with Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset & EDA", "🤖 Training & Evaluation", "🔮 Predict New Values", "🕓 History"])

# ==================== TAB 1: DATASET & EDA ====================
with tab1:
    st.header("📊 Dataset & Exploratory Data Analysis")
    st.dataframe(df.head(), use_container_width=True)
    st.info(f"**Total Records:** {df.shape[0]} rows | {df.shape[1]} columns")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Combined_MPG'], bins=30, kde=True, ax=ax, color='skyblue')
        ax.set_title('Combined MPG Distribution')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_ice = df[df['Vehicle_Category'] != 'EV'].copy()
        sns.scatterplot(data=df_ice, x='Combined_MPG', y='CO2_Emissions_g_per_mile', alpha=0.6, color='darkorange')
        ax.set_title('Combined MPG vs CO2 Emissions (ICE Only)')
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("💡 **Insight:** There is an inverse relationship between Combined MPG and CO2 Emissions. The higher the MPG, the lower the CO2 emissions produced.")

# ==================== TAB 2: TRAINING & EVALUATION ====================
with tab2:
    st.header("🤖 Model Training & Evaluation")

    if train_btn:
        df_reg = df[df['Vehicle_Category'] != 'EV'].copy()
        y = df_reg['CO2_Emissions_g_per_mile']
        X_full = df_reg[['Engine_Size_L', 'Engine_Cylinders', 'Combined_MPG']]

        X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=test_size, random_state=42)

        coefficients = {}
        if model_type.startswith("Simple"):
            X_train_s = X_train[['Engine_Size_L']]
            X_test_s = X_test[['Engine_Size_L']]
            model = LinearRegression()
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            st.success("✅ SLR model trained successfully!")
            coef_val = model.coef_[0]
            st.metric("Coefficient (Engine_Size_L)", f"{coef_val:.4f}")
            coefficients['Engine_Size_L'] = round(coef_val, 4)
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.success("✅ MLR model trained successfully!")
            for name, coef in zip(['Engine_Size_L', 'Engine_Cylinders', 'Combined_MPG'], model.coef_):
                st.metric(f"Coefficient {name}", f"{coef:.4f}")
                coefficients[name] = round(coef, 4)

        st.metric("Intercept", f"{model.intercept_:.4f}")

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("RMSE (g/mile)", f"{rmse:.4f}")

        # Save to session state
        st.session_state.model = model
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

        # Log to training history
        st.session_state.training_history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": model_type,
            "Test Size": test_size,
            "R² Score": round(r2, 4),
            "RMSE (g/mile)": round(rmse, 4),
            "Intercept": round(model.intercept_, 4),
            "Coefficients": str(coefficients),
        })

        # Plot Actual vs Predicted
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.6, color='darkorange')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        ax.set_xlabel('Actual CO2 (g/mile)')
        ax.set_ylabel('Predicted CO2 (g/mile)')
        ax.set_title(f'{model_type}: Actual vs Predicted')
        ax.legend()
        st.pyplot(fig)
    else:
        if st.session_state.model is None:
            st.info("Click **🚀 Train Model** in the sidebar to get started.")

# ==================== TAB 3: PREDICTION ====================
with tab3:
    st.header("🔮 Predict CO2 Emissions for a New Vehicle")
    st.info("Make sure you have trained a model first in the 'Training & Evaluation' tab.")

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            engine_size = st.number_input("Engine Size (L)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        with col2:
            cylinders = st.number_input("Number of Cylinders", min_value=3, max_value=16, value=4, step=1)
        with col3:
            combined_mpg = st.number_input("Combined MPG", min_value=5.0, max_value=150.0, value=25.0, step=1.0)

        submit = st.form_submit_button("🔍 Predict")

        if submit:
            if st.session_state.model is None:
                st.warning("⚠️ Please train a model first!")
            else:
                if model_type.startswith("Simple"):
                    input_data = pd.DataFrame([[engine_size]], columns=['Engine_Size_L'])
                else:
                    input_data = pd.DataFrame([[engine_size, cylinders, combined_mpg]],
                                              columns=['Engine_Size_L', 'Engine_Cylinders', 'Combined_MPG'])

                pred = st.session_state.model.predict(input_data)[0]
                st.success(f"🔮 Predicted CO2 Emission: **{pred:.2f} g/mile**")

                # Log to prediction history
                entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Model Used": model_type,
                    "Engine Size (L)": engine_size,
                    "Cylinders": cylinders if not model_type.startswith("Simple") else "N/A",
                    "Combined MPG": combined_mpg if not model_type.startswith("Simple") else "N/A",
                    "Predicted CO2 (g/mile)": round(pred, 2),
                }
                st.session_state.prediction_history.append(entry)

# ==================== TAB 4: HISTORY ====================
with tab4:
    st.header("🕓 Calculation History")

    st.subheader("🔮 Prediction History")
    if st.session_state.prediction_history:
        pred_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(pred_df, use_container_width=True)

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Prediction History (CSV)", csv, "prediction_history.csv", "text/csv")
        with col_clr:
            if st.button("🗑️ Clear Prediction History"):
                st.session_state.prediction_history = []
                st.rerun()
    else:
        st.info("No predictions made yet. Go to the **Predict New Values** tab to get started.")

    st.markdown("---")

    st.subheader("🤖 Training History")
    if st.session_state.training_history:
        train_df = pd.DataFrame(st.session_state.training_history)
        st.dataframe(train_df, use_container_width=True)

        col_dl2, col_clr2 = st.columns([1, 1])
        with col_dl2:
            csv2 = train_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Training History (CSV)", csv2, "training_history.csv", "text/csv")
        with col_clr2:
            if st.button("🗑️ Clear Training History"):
                st.session_state.training_history = []
                st.rerun()
    else:
        st.info("No training runs yet. Train a model first in the **Training & Evaluation** tab.")