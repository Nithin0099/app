import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the pre-trained model (you should save your trained model as a .pkl or .joblib file)
# For the sake of this example, let's simulate a basic model and vectorizer.
def load_model():
    # This is a placeholder function, and you should load your actual model and vectorizer
    # Example: return joblib.load('spam_model.pkl'), joblib.load('vectorizer.pkl')
    model = RandomForestClassifier()
    vectorizer = CountVectorizer()
    return model, vectorizer

# Placeholder trained data (for demonstration purposes only)
def train_model():
    # Dummy training data
    data = pd.DataFrame({
        'text': ['Free money', 'Hello, how are you?', 'Claim your prize now!', 'Hey, want to grab coffee?'],
        'label': [1, 0, 1, 0]  # 1: spam, 0: not spam
    })

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['label']
    
    # Train the model
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save model and vectorizer (in real-world scenario, you would save these models to disk)
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

# Uncomment to train and save the model (one-time only)
# train_model()

# Streamlit app
st.title("SMS Spam Detection")

st.write("Upload a CSV file containing SMS messages to predict if they're spam.")

# Upload a CSV file
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.error("The uploaded CSV file must contain a 'text' column with SMS messages.")
    else:
        # Load the pre-trained model and vectorizer
        model, vectorizer = load_model()

        # Vectorize the text data
        X_new = vectorizer.transform(df['text'])

        # Make predictions
        predictions = model.predict(X_new)

        # Add the prediction column to the dataframe
        df['prediction'] = ['Spam' if pred == 1 else 'Not Spam' for pred in predictions]

        # Show the predictions
        st.write("Predictions for the uploaded SMS messages:")
        st.write(df)

        # Optionally, allow the user to download the prediction results
        st.download_button("Download Predictions", df.to_csv(), file_name="predictions.csv", mime="text/csv")
