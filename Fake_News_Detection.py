import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = {
    "text": [
        "The government has announced a new stimulus package for small businesses.",
        "NASA confirms the presence of organic molecules on Mars.",
        "Aliens abducted a farmer in Nebraska last night.",
        "Cure for cancer found in a secret lab in the Himalayas.",
        "Elections to be held next month, says the Election Commission.",
        "Bill Gates caught developing weather-controlling machines.",
        "Doctors warn against viral TikTok health trend.",
        "Scientists create new material stronger than steel.",
        "Celebrity reveals she is actually a time traveler.",
        "Researchers discover microplastics in Arctic snow.",
        "Facebook to start charging for accounts next year.",
        "Breakthrough in battery tech could triple EV range.",
        "Man claims he lived on Mars for 3 years.",
        "City to install solar panels on all public buildings.",
        "Famous actor arrested for running secret spy agency.",
        "New vaccine shows 95% effectiveness in trials.",
        "AI becomes sentient, demands equal rights.",
        "Parliament passes new education reform bill.",
        "Study finds coffee may reduce risk of heart disease.",
        "UFO crash site discovered in the Amazon rainforest."
    ],
    "label": [
        "REAL", "REAL", "FAKE", "FAKE", "REAL",
        "FAKE", "REAL", "REAL", "FAKE", "REAL",
        "FAKE", "REAL", "FAKE", "REAL", "FAKE",
        "REAL", "FAKE", "REAL", "REAL", "FAKE"
    ]
}

df = pd.DataFrame(data)
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions, zero_division=0))

sample = "Scientists announce a new method to reverse aging."
print("Prediction:", pipeline.predict([sample])[0])
