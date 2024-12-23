import os

# Set the working directory
os.chdir("C:/Users/OITCOPowerR/OneDrive - Department of Veterans Affairs/Documents\Medallia Text Data Analysis")

# Verify the change
print("Current working directory:", os.getcwd())

import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Step 1: Load your data
file_path = "C:/Users/OITCOPowerR/OneDrive - Department of Veterans Affairs/Secure Messages Intercept_11.29.24.xlsx"
df = pd.read_excel(file_path)

# Ensure there are no missing values in relevant columns
text_column = "Why did you select that rating?"
tag_column = "Tag 1"
sentiment_column = "Positive/ negative/ mixed/ neutral"
df = df[[text_column, tag_column, sentiment_column]].dropna()

# Step 2: Train-Test Split for Topic Classification
X = df[text_column]
y = df[tag_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorize Text Data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train Topic Tagging Model
tag_classifier = RandomForestClassifier(random_state=42)
tag_classifier.fit(X_train_vec, y_train)

# Step 5: Predict Tags for Entire Dataset
df["Predicted Tag"] = tag_classifier.predict(vectorizer.transform(df[text_column]))

# Step 6: Sentiment Analysis
# Use pre-trained sentiment analysis from transformers
sentiment_analyzer = pipeline("sentiment-analysis")

# Predict sentiment for each comment
df["Predicted Sentiment"] = df[text_column].apply(
    lambda x: sentiment_analyzer(x)[0]["label"] if isinstance(x, str) else "Neutral"
)

# Step 7: Save Results to a New Excel File
output_file = "automated_tags_and_sentiment.xlsx"
df.to_excel(output_file, index=False)

print(f"Tagging and sentiment analysis completed. Results saved to {output_file}.")
