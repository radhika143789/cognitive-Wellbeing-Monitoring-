import streamlit as st
from datetime import datetime
import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Define mood options
mood_options = ["Happy", "Sad", "Angry", "Neutral", "Stressed", "Relaxed"]
mood_encoder = LabelEncoder().fit(mood_options)

# Initialize session state
if 'mood_log' not in st.session_state:
    st.session_state.mood_log = []

if 'cognitive_scores' not in st.session_state:
    st.session_state.cognitive_scores = []

# Function to encrypt data
def encrypt_data(data):
    return cipher_suite.encrypt(data.encode()).decode()

# Function to decrypt data
def decrypt_data(data):
    return cipher_suite.decrypt(data.encode()).decode()

# Function to log mood
def log_mood(mood, notes):
    st.session_state.mood_log.append({
        "timestamp": datetime.now(),
        "mood": encrypt_data(mood),
        "notes": encrypt_data(notes)
    })

# Function to log cognitive task score
def log_cognitive_score(score):
    st.session_state.cognitive_scores.append({
        "timestamp": datetime.now(),
        "score": encrypt_data(str(score))
    })

# Mood Tracking
st.title("Cognitive Wellbeing Monitoring")

st.header("Mood Tracking")
mood = st.selectbox("Select your mood", mood_options)
notes = st.text_area("Any additional notes?")
if st.button("Log Mood"):
    log_mood(mood, notes)
    st.success("Mood logged successfully!")

# Display mood log
if st.session_state.mood_log:
    st.subheader("Mood Log")
    decrypted_mood_log = [
        {
            "timestamp": entry["timestamp"],
            "mood": decrypt_data(entry["mood"]),
            "notes": decrypt_data(entry["notes"])
        }
        for entry in st.session_state.mood_log
    ]
    df_mood = pd.DataFrame(decrypted_mood_log)
    st.write(df_mood)

# Cognitive Task - Memory Game
st.header("Cognitive Task: Memory Game")
if st.button("Start Memory Game"):
    numbers = [random.randint(1, 100) for _ in range(5)]
    st.write(f"Remember these numbers: {numbers}")
    st.session_state.memory_numbers = numbers

if 'memory_numbers' in st.session_state:
    user_numbers = st.text_input("Enter the numbers you remember, separated by commas")
    if st.button("Submit"):
        try:
            user_numbers = list(map(int, user_numbers.split(',')))
            score = sum([1 for i, j in zip(user_numbers, st.session_state.memory_numbers) if i == j])
            log_cognitive_score(score)
            st.success(f"You remembered {score} numbers correctly!")
        except ValueError:
            st.error("Please enter numbers separated by commas")

# Display cognitive scores
if st.session_state.cognitive_scores:
    st.subheader("Cognitive Task Scores")
    decrypted_scores = [
        {
            "timestamp": entry["timestamp"],
            "score": decrypt_data(entry["score"])
        }
        for entry in st.session_state.cognitive_scores
    ]
    df_scores = pd.DataFrame(decrypted_scores)
    st.write(df_scores)

# Simple Mental Health Questionnaire
st.header("Mental Health Questionnaire")
q1 = st.slider("How often have you felt anxious over the past week?", 0, 10, 5)
q2 = st.slider("How often have you felt depressed over the past week?", 0, 10, 5)
q3 = st.slider("How often have you felt stressed over the past week?", 0, 10, 5)

if st.button("Submit Questionnaire"):
    st.success("Questionnaire submitted!")
    st.write(f"Anxiety level: {q1}, Depression level: {q2}, Stress level: {q3}")

# Machine Learning Model to Predict Mood
st.header("Mood Prediction")
if st.button("Predict Mood"):
    if len(st.session_state.mood_log) >= 2:
        df = pd.DataFrame(decrypted_mood_log)
        df['mood'] = mood_encoder.transform(df['mood'])
        X = df[['timestamp']].apply(lambda x: x.timestamp()).values.reshape(-1, 1)
        y = df['mood'].values

        model = LinearRegression().fit(X, y)
        next_timestamp = datetime.now().timestamp()
        prediction = model.predict(np.array([[next_timestamp]]))
        predicted_mood = mood_encoder.inverse_transform([int(prediction[0])])[0]

        st.success(f"Predicted mood for the next entry: {predicted_mood}")
    else:
        st.error("Not enough data to make a prediction")

st.sidebar.header("Options")
if st.sidebar.button("Reset Data"):
    st.session_state.mood_log = []
    st.session_state.cognitive_scores = []
    st.success("Data reset successfully!")
