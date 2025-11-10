import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text) # removes puncuation
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)