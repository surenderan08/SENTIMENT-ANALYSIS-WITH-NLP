import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the Dataset
# Replace with your own dataset if needed
# Example: Customer reviews dataset
reviews = {
    "text": [
        "I love this product! It's amazing!",
        "Terrible experience, I hate it.",
        "Decent quality, but could be better.",
        "Absolutely fantastic! Highly recommend.",
        "Not great, not terrible. Just okay.",
        "Worst purchase ever!",
        "Pretty good value for the price.",
        "This is the best thing I've ever bought!",
        "Awful quality, do not buy.",
        "Satisfied with the purchase."
    ],
    "sentiment": [1, 0, 1, 1, 1, 0, 1, 1, 0, 1]  # 1: Positive, 0: Negative
}
df = pd.DataFrame(reviews)

# Step 2: Preprocessing
X = df["text"]
y = df["sentiment"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 5: Predict and Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Optional: Example Prediction
example_review = ["This product is outstanding!"]
example_tfidf = tfidf_vectorizer.transform(example_review)
predicted_sentiment = model.predict(example_tfidf)
print(f"Example Review Sentiment: {'Positive' if predicted_sentiment[0] == 1 else 'Negative'}")
