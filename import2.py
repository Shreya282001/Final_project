import os
import streamlit as st
import requests
import transformers
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from app2 import preprocess_text
from utils.b2 import B2
from dotenv import load_dotenv
# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')




# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()

REMOTE_DATA = 'WASHINGTON (CNN)1.csv'
# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_keyID'],
        secret_key=os.environ['B2_applicationKey'])
@st.cache_data
def get_data():
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df= b2.get_df(REMOTE_DATA)


preprocessed_articles = []
for article_text in df['article']:
    preprocessed_articles.append(preprocess_text(article_text))

preprocessed_texts = [' '.join(sentences) for sentences in preprocessed_articles]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
feature_names = vectorizer.get_feature_names_out()

# Load your extractive summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Function to fetch article content from URL
def fetch_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = " ".join([p.get_text() for p in soup.find_all('p')])
        return article_text
    except:
        st.error("Error fetching article content. Please check the URL.")

# Streamlit app
def main():
    st.title("CNN Article Summarizer")
    st.write("This app summarizes CNN articles using extractive summarization.")

    article_url = st.text_input("Enter CNN Article URL:")
    
    if st.button("Summarize"):
        if article_url:
            article_text = fetch_article_content(article_url)
            if article_text:
                preprocessed_article = preprocess_text(article_text)
                tfidf_scores = [sum(tfidf_matrix[0, vectorizer.vocabulary_[word]] for word in sentence.split() if word in vectorizer.vocabulary_) for sentence in preprocessed_article]
                top_sentence_indices = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)[:3]
                top_sentences = [preprocessed_article[j] for j in top_sentence_indices]
                summarized_article = ' '.join(top_sentences)
                st.subheader("Summary:")
                st.write(summarized_article)
        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()
