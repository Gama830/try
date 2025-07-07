import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to load and preprocess data
def load_data(file):
    df = pd.read_csv(file)
    # Replace NaNs in the 'text' column with empty strings
    df['text'].fillna('', inplace=True)
    return df

# Function to train the model
def train_model(df):
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy

# Function to make predictions
def predict_news(model, vectorizer, news):
    news_tfidf = vectorizer.transform([news])
    prediction = model.predict(news_tfidf)
    return prediction[0]

# Streamlit app
def main():
    st.title("Fake News Detector")

    st.write("Upload a CSV file with news text and labels (0: True, 1: Fake).")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model, vectorizer, accuracy = train_model(df)
                st.session_state['model'] = model
                st.session_state['vectorizer'] = vectorizer
                st.session_state['accuracy'] = accuracy
                st.success(f"Model trained with accuracy: {accuracy:.2f}")

    if 'model' in st.session_state and 'vectorizer' in st.session_state:
        st.write("Enter news text to predict if it is true or fake:")

        news_text = st.text_area("News Text")

        if st.button("Predict"):
            if news_text.strip():
                prediction = predict_news(st.session_state['model'], st.session_state['vectorizer'], news_text)
                st.write(f"The news is predicted to be: {'Fake' if prediction == 1 else 'True'}")
            else:
                st.warning("Please enter some news text to predict.")

if __name__ == "__main__":
    main()
