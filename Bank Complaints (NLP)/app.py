import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from db_utils import insert_complaint 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load your saved NLP model
with open('/Users/sahnoontm/Documents/ML/NLP/bank_complaints/nlp_model.pkl', 'rb') as f:
    nlp_model = pickle.load(f)

model = nlp_model['model']
vectorizer = nlp_model['vectorizer']
label_encoder = nlp_model['label_encoder']

st.title("🔍 Bank Complaint Category Predictor")
st.markdown("Enter a customer's complaint, and the model will predict the category.")

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def text_preprocessing(text):
    text = text.lower()
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    lemmatized_words = []
    for word, tag in pos_tags:
        if word.isalpha() and word not in stop_words:
            pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos)
            lemmatized_words.append(lemma)
    return ' '.join(lemmatized_words)

user_input = st.text_area("✉️ Enter Complaint Text:")


if st.button("🔮 Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter a complaint before predicting.")
    else:

        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)
        predicted_category = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🏷️ Predicted Category: **{predicted_category}**")
        success = insert_complaint(user_input, predicted_category)
        if success:
            st.info("✅ Complaint saved to database.")
        else:
            st.error("❌ Failed to save to database.")
