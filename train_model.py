import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('url_dataset.csv')  # Ensure your dataset has 'url' and 'label' columns

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['url'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline that vectorizes the URLs and then applies a Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Test the model
predicted = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predicted)}')

# Save the model
joblib.dump(model, 'url_model.pkl')
