# model/train_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from utils.preprocess import clean_text

# 1. Load data
data = pd.read_csv("data/cleaned_data.csv", encoding="utf-8")
data = data[["label", "text"]]

# 2. Preprocess
data["clean_text"] = data["text"].apply(clean_text)
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# 3. Split (stratified for balanced spam/ham)
x_train, x_test, y_train, y_test = train_test_split(
    data["clean_text"], data["label_num"],
    test_size=0.2, random_state=42, stratify=data["label_num"]
)

# 4. TF-IDF Vectorizer (enhanced)
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# 5. Train model
model = MultinomialNB(alpha=0.1)
model.fit(x_train_tfidf, y_train)

# 6. Evaluate
y_pred = model.predict(x_test_tfidf)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model and vectorizer
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")