import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


st.title("🛡️ AI Insurance Claim Fraud Detection")


@st.cache_data
def load_data():

    df = pd.read_csv("insurance_claims.csv")

    df = df.drop(
    [
    "policy_number",
    "insured_zip",
    "policy_bind_date",
    "incident_date",
    "incident_location"
    ],
    axis=1,
    errors="ignore"
    )

    return df


df = load_data()


# Encode categorical columns

for col in df.select_dtypes(include="object").columns:

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# Target

X = df.drop("fraud_reported",axis=1)

y = df["fraud_reported"]


X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)


accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)


st.success(
f"Model Accuracy: {accuracy*100:.2f}%"
)


st.header("Enter Claim Details")


inputs=[]

for col in X.columns:

    value = st.number_input(
        col,
        value=float(X[col].mean())
    )

    inputs.append(value)



if st.button("Predict Fraud"):

    result = model.predict(
        np.array(inputs).reshape(1,-1)
    )


    if result[0]==1:

        st.error("⚠️ Fraud Claim Detected")

    else:

        st.success("✅ Genuine Claim")