from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Function to train and save the model
def train_model():
    # Check if the model already exists to avoid re-training
    if not os.path.exists('url_model.pkl'):
        # Load the dataset
        data = pd.read_csv('url_dataset.csv')  # Ensure your dataset has 'url' and 'label' columns
        
        # Check if the required columns are present
        print("Columns in dataset:", data.columns)
        if 'url' not in data.columns or 'label' not in data.columns:
            raise KeyError("Dataset must contain 'url' and 'label' columns")

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(data['url'], data['label'], test_size=0.2, random_state=42)

        # Create a pipeline that vectorizes the URLs and then applies a Naive Bayes classifier
        model = make_pipeline(CountVectorizer(), MultinomialNB())

        # Train the model
        model.fit(X_train, y_train)

        # Test the model
        predicted = model.predict(X_test)
        accuracy = accuracy_score(y_test, predicted)

        # Save the model
        joblib.dump(model, 'url_model.pkl')

# Train the model if not already trained
train_model()

# Load the trained model
model = joblib.load('url_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


#The /predict route is a REST API endpoint that handles POST requests.
#When a user submits a URL through the form on the home page, the form sends a POST request to this endpoint.
#The endpoint retrieves the submitted URL from the request, uses the loaded model to make a prediction, and then renders the result.html template with the prediction results.
@app.route('/predict', methods=['POST'])

#Breakdown of the API PARTS: This decorator defines a new route for the Flask application. The route /predict is designed to handle POST requests.

def predict():
    if request.method == 'POST':
        url = request.form['url']
        prediction = model.predict([url])[0]
        return render_template('result.html', prediction=prediction, url=url)

if __name__ == '__main__':
    app.run(debug=True)


#Here, we initialize a Flask application instance (app). Flask is a web framework for Python that allows us to build web applications and create API endpoints.
#from flask import Flask, render_template, request
#import joblib
#import os

app = Flask(__name__)
