import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset into a DataFrame
dataset_path = r'c:\Users\SHREYA\OneDrive\Documents\MVP\WASHINGTON (CNN).xlsx'
df = pd.read_excel(dataset_path)

# Preprocess the dataset
def preprocess_text(article_text):
    sentences = sent_tokenize(article_text)
    sentences = [re.sub('<.*?>', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', sentence).lower() for sentence in sentences]
    stop_words = set(stopwords.words('english'))
    sentences = [' '.join(word for word in sentence.split() if word not in stop_words) for sentence in sentences]
    return sentences

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
