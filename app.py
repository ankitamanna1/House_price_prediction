import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    area = int(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories=int(request.form['stories'])
    parks=int(request.form['parking'])

    # Combine the inputs into an array for the model
    input_features = np.array([[area, bedrooms, bathrooms,stories,parks]])

    # Perform prediction
    prediction = model.predict(input_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='House price should be ${}'.format(output))

if __name__ == "__main__":
    app.run(port=8000, debug=True)
