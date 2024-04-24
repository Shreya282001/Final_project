import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(article_text):
    sentences = sent_tokenize(article_text)
    sentences = [re.sub('<.*?>', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', sentence).lower() for sentence in sentences]
    stop_words = set(stopwords.words('english'))
    sentences = [' '.join(word for word in sentence.split() if word not in stop_words) for sentence in sentences]
    return sentences
