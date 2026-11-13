import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# 2. Load Model and Dataset
folder = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(folder, "loan_approval_model.pkl"))
df = pd.read_csv(os.path.join(folder, "Loan_approval_dataset.csv"))
df.columns = df.columns.str.strip()

# 3. Project Title
st.title("🏦 Loan Approval Prediction System")
st.write("Predict loan approval using Machine Learning.")

# 4. Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Compare Models", "Predict Loan", "Explore Data"
])

# 5. Overview
with tab1:
    st.subheader("Loan Application Overview")

    status = df["loan_status"].astype(str).str.strip()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applications", len(df))
    c2.metric("Approved Loans", (status == "Approved").sum())
    c3.metric("Rejected Loans", (status == "Rejected").sum())

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x=status, ax=ax)
    ax.set_title("Loan Status Distribution")
    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Number of Applications")
    st.pyplot(fig)
    plt.close(fig)

# 6. Compare Models
with tab2:
    st.subheader("Model Performance Comparison")

    results = pd.DataFrame({
        "Model": [
            "XGBoost", "Random Forest", "Decision Tree",
            "SVM", "Logistic Regression", "KNN"
        ],
        "Accuracy": [
            0.981265, 0.981265, 0.973068,
            0.944965, 0.922717, 0.900468
        ],
        "Precision": [
            0.984227, 0.990415, 0.971698,
            0.942308, 0.915858, 0.886364
        ],
        "Recall": [
            0.965944, 0.959752, 0.956656,
            0.910217, 0.876161, 0.845201
        ],
        "F1 Score": [
            0.975000, 0.974843, 0.964119,
            0.925984, 0.895570, 0.865293
        ]
    })

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox(
        "Select Metric",
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=results.sort_values(metric, ascending=False),
        x=metric,
        y="Model",
        ax=ax
    )
    ax.set_xlim(0, 1)
    ax.set_title(f"Model Comparison - {metric}")
    st.pyplot(fig)
    plt.close(fig)

    st.success("Best Model: XGBoost (Highest F1 Score)")

# 7. Predict Loan
with tab3:
    st.subheader("Applicant Details")

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            dependents = st.number_input(
                "Number of Dependents", min_value=0, max_value=10, value=0
            )
            education = st.selectbox(
                "Education", ["Graduate", "Not Graduate"]
            )
            employed = st.selectbox(
                "Self Employed", ["No", "Yes"]
            )
            income = st.number_input(
                "Annual Income", min_value=0, value=5000000
            )
            loan = st.number_input(
                "Loan Amount", min_value=0, value=10000000
            )
            term = st.number_input(
                "Loan Term (Months)", min_value=1, value=12
            )

        with col2:
            cibil = st.number_input(
                "CIBIL Score", min_value=300, max_value=900, value=700
            )
            residential = st.number_input(
                "Residential Assets Value", min_value=0, value=1000000
            )
            commercial = st.number_input(
                "Commercial Assets Value", min_value=0, value=500000
            )
            luxury = st.number_input(
                "Luxury Assets Value", min_value=0, value=500000
            )
            bank = st.number_input(
                "Bank Asset Value", min_value=0, value=500000
            )

        submit = st.form_submit_button(
            "Predict Loan Status", use_container_width=True
        )

    if submit:
        input_data = pd.DataFrame([{
            "no_of_dependents": dependents,
            "education": " " + education,
            "self_employed": " " + employed,
            "income_annum": income,
            "loan_amount": loan,
            "loan_term": term,
            "cibil_score": cibil,
            "residential_assets_value": residential,
            "commercial_assets_value": commercial,
            "luxury_assets_value": luxury,
            "bank_asset_value": bank
        }])

        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        # LabelEncoder mapping used in the training notebook:
        # 0 = Approved, 1 = Rejected
        if prediction == 0:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.caption(
            "Prediction is generated by the trained XGBoost model. "
            "This is an educational project, not a lending decision."
        )

# 8. Explore Data
with tab4:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_data = df.select_dtypes(include="number")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)
