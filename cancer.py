import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# === Streamlit Title ===
st.title("🎗️ Breast Cancer Prediction App")

# === Step 1: Load and Train Model Automatically ===
st.write("### 🧠 Training Model Automatically Using 'breast-cancer.csv'")

try:
    # Load your dataset (must be in the same folder)
    data_df = pd.read_csv("breast-cancer.csv")

    # Encode non-numeric columns
    for col in data_df.columns:
        if data_df[col].dtype == 'object' and col != 'target':
            data_df[col] = pd.Categorical(data_df[col]).codes

    # Encode target if text
    if data_df['target'].dtype == 'object':
        data_df['target'] = pd.Categorical(data_df['target']).codes

    # Split data
    X = data_df.drop('target', axis=1)
    y = data_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    # === Train Decision Tree Model ===
    DT = DecisionTreeClassifier(
        criterion='entropy',   # You can also use 'gini'
        max_depth=6,           # To prevent overfitting
        random_state=42
    )
    DT.fit(X_train, y_train)

    # Model accuracy
    y_pred = DT.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

except Exception as e:
    st.error(f"❌ Error loading or training model: {e}")
    st.stop()

# === Step 2: Input Features for Prediction ===
st.header("🔍 Enter Patient Details for Prediction")

user_input = {}
for col in X.columns:
    value = st.number_input(f"Enter value for **{col}**:", value=0.0, format="%.4f")
    user_input[col] = value

# === Step 3: Predict Button ===
if st.button("🔮 Predict"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = DT.predict(input_df)[0]

        # Get probability for class 1 (cancer)
        cancer_index = list(DT.classes_).index(1) if 1 in DT.classes_ else 0
        prob = DT.predict_proba(input_df)[0][cancer_index]

        accuracy_percent = round(accuracy * 100, 2)

        # Risk categorization
        if prob < 0.33:
            risk_level = "🟢 Low Risk"
            color = "#ccffcc"
            text_color = "#006600"
            message = "No major signs of breast cancer detected."
        elif prob < 0.67:
            risk_level = "🟡 Medium Risk"
            color = "#fff3cd"
            text_color = "#996600"
            message = "Some patterns may indicate possible concerns. Medical review suggested."
        else:
            risk_level = "🔴 High Risk"
            color = "#ffcccc"
            text_color = "#b30000"
            message = "High probability of breast cancer. Please consult a doctor immediately."

        # === Stylish Result Box ===
        st.markdown(f"""
        <div style='background-color:{color};padding:25px;border-radius:15px;'>
            <h2 style='color:{text_color};text-align:center;'>🩺 Diagnosis Report</h2>
            <hr>
            <h3 style='color:{text_color};'>Prediction Result:</h3>
            <p style='font-size:18px;color:{text_color};'>
                {'Breast Cancer Detected' if prediction == 1 else 'No Breast Cancer Detected'}
            </p>
            <h3 style='color:{text_color};'>Model Accuracy:</h3>
            <p style='font-size:18px;color:{text_color};'>{accuracy_percent}%</p>
            <h3 style='color:{text_color};'>Risk Level:</h3>
            <p style='font-size:18px;color:{text_color};'>{risk_level}</p>
            <hr>
            <p style='font-size:16px;color:{text_color};'>{message}</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
