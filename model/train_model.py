import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from utils.preprocess import clean_text


data = pd.read_csv("data\cleaned_data.csv", encoding="utf-8")
#print(data.head())

data = data[["label", "text"]]


# Preprocess
data["clean_text"] = data["text"].apply(clean_text)
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# Train Test data split
x_train, x_test, y_train, y_test = train_test_split(
    data['clean_text'], data['label_num'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=3000)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.fit_transform(x_test)


# Training the model

model = MultinomialNB()
model.fit(x_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(x_test_tfidf)
print("Accuracy ", accuracy_score(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))

# Saving the model for future UI use
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully")