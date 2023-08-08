from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# load the trained model
model = pickle.load(open('models/pipe3.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the input values from the form
    name = request.form['name']
    brand = request.form['brand']
    max_power = float(request.form['max_power'])
    max_torque = float(request.form['max_torque'])
    fuel_tank_capacity = float(request.form['fuel_tank_capacity'])
    top_speed = float(request.form['top_speed'])
    front_brake_type = request.form['front_brake_type']
    kerb_weight = float(request.form['kerb_weight'])
    overall_length = float(request.form['overall_length'])
    overall_width = float(request.form['overall_width'])
    wheelbase = float(request.form['wheelbase'])
    overall_height = float(request.form['overall_height'])

    # Check for zero or negative inputs
    if any(val <= 0 for val in [max_power, max_torque, fuel_tank_capacity, top_speed, kerb_weight, overall_length, overall_width, wheelbase, overall_height]):
        return render_template('error.html', message="Please enter valid positive values for all parameters.")

    # make a prediction using the loaded model
    input_data = [[name, brand, max_power, max_torque, fuel_tank_capacity, top_speed, front_brake_type, kerb_weight, overall_length, overall_width, wheelbase, overall_height]]
    prediction = model.predict(input_data)[0]

    # return the prediction as a JSON response
    response = {'price': prediction}
    return render_template('result.html', price=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
