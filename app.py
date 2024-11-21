from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import dataset  # Import dataset module

app = Flask(__name__)

# Load and prepare dataset
data = dataset.generate_dataset()

# Split the data into features and target
X = data[['discount', 'quantity']]
y = data['feedback']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Home route (Feedback Predictor)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission for prediction
@app.route('/predict', methods=['POST'])
def predict():
    customer_name = request.form['customer_name']
    ship_mode = request.form['ship_mode']
    category = request.form['category']
    product_name = request.form['product_name']
    discount = float(request.form['discount'])
    quantity = int(request.form['quantity'])

    input_data = pd.DataFrame({
        'discount': [discount],
        'quantity': [quantity]
    })

    prediction = model.predict(input_data)
    predicted_stars = prediction[0]

    return render_template('result.html', customer_name=customer_name, predicted_stars=predicted_stars)

# Route for DMart Dashboard page
@app.route('/dmartdashboard')
def dmart_dashboard():
    return render_template('dmart_dashboard.html')

# Route for Comparative Analysis page
@app.route('/comparativeanalysis')
def comparative_analysis():
    return render_template('comparative_analysis.html')

if __name__ == "__main__":
    app.run(debug=True)
