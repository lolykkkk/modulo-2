# ---- Librerias ----
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier 

# ---- Archivo y Preprocesamiento ----
train = pd.read_csv("train_ctrUa4K.csv")
train["Is_female"] = train["Gender"].map({"Male":0, "Female":1})
train["Is_married"] = train["Married"].map({"No":0, "Yes":1})
train["Loan_Status"] = train["Loan_Status"].map({"N":0, "Y":1})
train.drop(columns = ["Gender", "Married"], inplace = True)
train = train.dropna()
X = train[["Is_female", "Is_married", "ApplicantIncome", "LoanAmount", "Credit_History"]]
y = train[["Loan_Status"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)

def prediction(Is_female, Is_married, ApplicantIncome, LoanAmount, Credit_History):
    # Pre procesamiento
    # Is_female
    if Is_female == "M":
        Is_female = 0
    else:
        Is_female = 1 
    # Is_married
    if Is_married == "Casado/a":
        Is_married = 1
    else:
        Is_married = 0 
    # LoanAmount
    LoanAmount /= 1000
    # Credit_History
    if Credit_History == "Sí":
        Credit_History = 1
    else:
        Credit_History = 0  
    # Hacemos predicción
    pred = model.predict([[Is_female, Is_married, ApplicantIncome, LoanAmount, Credit_History]])
    if pred == 0:
        pred = "Denegado"
    else:
        pred = "Concedido"
    return pred

# Parámetros de entrada
Is_female = st.selectbox("Género", ("M", "F"))
Is_married = st.selectbox("Estado Civil", ("Casado/a", "Soltero/a"))
ApplicantIncome = st.number_input("Tu salario")
LoanAmount = st.number_input("Valor de tú préstamo")
Credit_History = st.selectbox("Deudas", ("Sí", "No"))

# Botón de predicción
if st.button("¿Obtendrás tu préstamo?"):
    result = prediction(Is_female, Is_married, ApplicantIncome, LoanAmount, Credit_History)
    if result == "Concedido":
        st.success("¡Tu préstamo ha sido concedido!")
    else:
        st.success("¡Prueba otra vez el próximo año!")


