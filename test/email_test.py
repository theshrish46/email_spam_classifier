import pickle

# Load saved model & vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

test_emails = [
    "Congratulations! You've won a free iPhone 15. Click here to claim your reward!",
    "Hey, are we still meeting at 3 PM for the project discussion?",
    "Job Alert! Free High salary for students. Apply here for free registration.",
    "Your Amazon order has been shipped and will arrive tomorrow.",
    "Earn ₹20,000 per week working from home! Apply now."
]

# Transform and predict
X = vectorizer.transform(test_emails)
preds = model.predict(X)

for email, label in zip(test_emails, preds):
    print(f"\nEMAIL: {email}\nPREDICTION: {'SPAM' if label==1 else 'HAM'}")
