from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the data
data = pd.read_csv('cardio_train.csv')

# Load the model and the scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html', data=data.to_html(index=False))

# @app.route('/data', methods=['GET'])
# def get_data():
#     return jsonify(data.to_dict(orient='records'))

# @app.route('/add', methods=['POST'])
# def add_data():
#     new_data = request.json
#     new_row = pd.DataFrame([new_data])
#     global data
#     data = pd.concat([data, new_row], ignore_index=True)
#     data.to_csv('data.csv', index=False)
#     return jsonify({'message': 'Data added successfully'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=False)
