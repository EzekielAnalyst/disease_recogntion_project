# import streamlit as st
import pandas as pd
import numpy as np
import pickle
import csv
from sklearn import preprocessing
from fuzzywuzzy import process
import streamlit as st

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as file:
    clf = pickle.load(file)

# Load the label encoder
training = pd.read_csv('Training.csv')
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)

# Load additional data
def getSeverityDict():
    severityDictionary = {}
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 2:
                continue  # Skip malformed rows
            severityDictionary[row[0]] = int(row[1])
    return severityDictionary

def getDescription():
    description_list = {}
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 2:
                continue  # Skip malformed rows
            description_list[row[0]] = row[1]
    return description_list

def getprecautionDict():
    precautionDictionary = {}
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 5:
                continue  # Skip malformed rows
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    return precautionDictionary

severityDictionary = getSeverityDict()
description_list = getDescription()
precautionDictionary = getprecautionDict()

# Function to get the closest matching symptoms using fuzzywuzzy
def get_closest_symptoms(input_symptoms, all_symptoms, threshold=80):
    matched_symptoms = []
    for symptom in input_symptoms:
        matches = process.extract(symptom, all_symptoms, limit=5)
        for match in matches:
            if match[1] >= threshold:
                matched_symptoms.append(match[0])
    return matched_symptoms

# Initialize session state variables
if 'matched_symptoms' not in st.session_state:
    st.session_state.matched_symptoms = []
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = {}

# Streamlit App
st.title("Uganda Christian University Healthcare Diagnosis Assistant Project")

st.sidebar.subheader("Hillary Luboyera")
st.sidebar.subheader("Serunjogi Ezekiel Mulondo")
st.sidebar.subheader("Andrew Angenrwot")



st.subheader("Enter your symptoms")
symptoms_input = st.text_area("Type your symptoms separated by commas")
days_input = st.number_input("Number of days experiencing symptoms", min_value=1, max_value=100, step=1, value=1)

if st.button("Find Matching Symptoms"):
    if symptoms_input.strip():
        input_symptoms = [symptom.strip() for symptom in symptoms_input.split(',')]
        all_symptoms = training.columns[:-1]
        st.session_state.matched_symptoms = get_closest_symptoms(input_symptoms, all_symptoms)

        if st.session_state.matched_symptoms:
            st.session_state.selected_symptoms = {symptom: False for symptom in st.session_state.matched_symptoms}
    else:
        st.write("Please enter at least one symptom.")

if st.session_state.matched_symptoms:
    st.write("Select the symptoms you are experiencing:")
    for symptom in st.session_state.matched_symptoms:
        st.session_state.selected_symptoms[symptom] = st.checkbox(symptom, value=st.session_state.selected_symptoms[symptom])

if st.button("Diagnose"):
    confirmed_symptoms = [symptom for symptom, selected in st.session_state.selected_symptoms.items() if selected]
    if confirmed_symptoms:
        input_vector = np.zeros(len(training.columns) - 1)
        total_severity = 0
        for symptom in confirmed_symptoms:
            if symptom in training.columns:
                input_vector[training.columns.get_loc(symptom)] = 1
            total_severity += severityDictionary.get(symptom, 0)

        prediction = clf.predict([input_vector])
        disease = le.inverse_transform(prediction)[0]

        severity_score = (total_severity * days_input) / len(confirmed_symptoms)

        st.write(f"The predicted disease is: {disease}")
        st.write(f"Description: {description_list.get(disease, 'No description available')}")
        st.write("Precautions:")
        precautions = precautionDictionary.get(disease, [])
        for i, precaution in enumerate(precautions):
            st.write(f"{i + 1}. {precaution}")
        
        st.write(f"Total Severity Score: {severity_score:.2f}")
        if severity_score > 13:
            st.write("Condition: Critical, needs immediate attention")
        else:
            st.write("Condition: Might not be critical but should take precautions")
    else:
        st.write("Please select at least one symptom for diagnosis.")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import csv
# from sklearn import preprocessing
# from fuzzywuzzy import process
# from pathlib import Path

# # Load the trained model
# model_path = Path('decision_tree_model.pkl')
# with model_path.open('rb') as file:
#     clf = pickle.load(file)

# # Load the label encoder
# training_path = Path('Training.csv')
# training = pd.read_csv(training_path)
# y = training['prognosis']
# le = preprocessing.LabelEncoder()
# le.fit(y)

# # Load additional data
# def getSeverityDict():
#     severityDictionary = {}
#     file_path = Path('symptom_severity.csv')
#     with file_path.open() as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for row in csv_reader:
#             if len(row) < 2:
#                 continue  # Skip malformed rows
#             severityDictionary[row[0]] = int(row[1])
#     return severityDictionary

# def getDescription():
#     description_list = {}
#     file_path = Path('symptom_Description.csv')
#     with file_path.open() as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for row in csv_reader:
#             if len(row) < 2:
#                 continue  # Skip malformed rows
#             description_list[row[0]] = row[1]
#     return description_list

# def getprecautionDict():
#     precautionDictionary = {}
#     file_path = Path('symptom_precaution.csv')
#     with file_path.open() as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for row in csv_reader:
#             if len(row) < 5:
#                 continue  # Skip malformed rows
#             precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
#     return precautionDictionary

# severityDictionary = getSeverityDict()
# description_list = getDescription()
# precautionDictionary = getprecautionDict()

# # Function to get the closest matching symptoms using fuzzywuzzy
# def get_closest_symptoms(input_symptoms, all_symptoms, threshold=80):
#     matched_symptoms = []
#     for symptom in input_symptoms:
#         matches = process.extract(symptom, all_symptoms, limit=5)
#         for match in matches:
#             if match[1] >= threshold:
#                 matched_symptoms.append(match[0])
#     return matched_symptoms

# # Initialize session state variables
# if 'matched_symptoms' not in st.session_state:
#     st.session_state.matched_symptoms = []
# if 'selected_symptoms' not in st.session_state:
#     st.session_state.selected_symptoms = {}

# # Streamlit App
# st.title("Uganda Christian University Healthcare Diagnosis Assistant Project")

# st.sidebar.subheader("Hillary Luboyera")
# st.sidebar.subheader("Serunjogi Ezekiel Mulondo")
# st.sidebar.subheader("Andrew Angenrwot")

# st.subheader("Enter your symptoms")
# symptoms_input = st.text_area("Type your symptoms separated by commas")
# days_input = st.number_input("Number of days experiencing symptoms", min_value=1, max_value=100, step=1, value=1)

# if st.button("Find Matching Symptoms"):
#     if symptoms_input.strip():
#         input_symptoms = [symptom.strip() for symptom in symptoms_input.split(',')]
#         all_symptoms = training.columns[:-1]
#         st.session_state.matched_symptoms = get_closest_symptoms(input_symptoms, all_symptoms)

#         if st.session_state.matched_symptoms:
#             st.session_state.selected_symptoms = {symptom: False for symptom in st.session_state.matched_symptoms}
#     else:
#         st.write("Please enter at least one symptom.")

# if st.session_state.matched_symptoms:
#     st.write("Select the symptoms you are experiencing:")
#     for symptom in st.session_state.matched_symptoms:
#         st.session_state.selected_symptoms[symptom] = st.checkbox(symptom, value=st.session_state.selected_symptoms[symptom])

# if st.button("Diagnose"):
#     confirmed_symptoms = [symptom for symptom, selected in st.session_state.selected_symptoms.items() if selected]
#     if confirmed_symptoms:
#         input_vector = np.zeros(len(training.columns) - 1)
#         total_severity = 0
#         for symptom in confirmed_symptoms:
#             if symptom in training.columns:
#                 input_vector[training.columns.get_loc(symptom)] = 1
#             total_severity += severityDictionary.get(symptom, 0)

#         prediction = clf.predict([input_vector])
#         disease = le.inverse_transform(prediction)[0]

#         severity_score = (total_severity * days_input) / len(confirmed_symptoms)

#         st.write(f"The predicted disease is: {disease}")
#         st.write(f"Description: {description_list.get(disease, 'No description available')}")
#         st.write("Precautions:")
#         precautions = precautionDictionary.get(disease, [])
#         for i, precaution in enumerate(precautions):
#             st.write(f"{i + 1}. {precaution}")
        
#         st.write(f"Total Severity Score: {severity_score:.2f}")
#         if severity_score > 13:
#             st.write("Condition: Critical, needs immediate attention")
#         else:
#             st.write("Condition: Might not be critical but should take precautions")
#     else:
#         st.write("Please select at least one symptom for diagnosis.")
