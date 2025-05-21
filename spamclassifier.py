import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', header=None)
    df.columns = ['label', 'message']
    return df

# Train model
@st.cache_resource
def train_model():
    df = load_data()
    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 and 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# Streamlit App
st.title("📧 Spam Email Classifier")
st.write("Enter a message below to check if it's **Spam** or **Not Spam**")

model, acc = train_model()

user_input = st.text_area("Enter the email text here:", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([user_input])[0]
        label = "Spam 🚫" if prediction else "Not Spam ✅"
        st.subheader("Result:")
        st.success(label)

st.markdown("---")
st.caption(f"Model Accuracy: {acc:.2%}")
