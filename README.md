# Custom Bike Pricing Optimizer using Machine Learning

This project aims to provide a machine learning solution for predicting the price of custom-built motorcycles based on their various features such as bike model name, brand, max power, max torque, fuel tank capacity, top speed, front brake type, kerb weight, overall length, overall width, wheelbase, and overall height. It aims to make decision making of motercycle manufacturing companies easier for next model update. It utilizes several regression algorithms and data preprocessing techniques to optimize the accuracy of the predictions.

## Installation 

1. Clone the repository from GitHub:
```bash git clone https://github.com/Omkar-Rajkumar-Khade/Custom_Bike_Pricing_Optimizer.git ```

2. Install the required dependencies using pip:
`pip install -r requirements.txt`


## Getting Started

## How to Use
To get started with this project, follow these steps:
* Clone this repository
* Install the required dependencies using pip: pip install -r requirements.txt
* Start the Flask server: python app.py
* Send a POST request to the /predict endpoint with a JSON payload containing the values for bike model name, brand, max power, max torque, fuel tank capacity, top speed, front brake type, kerb weight, overall length, overall width, wheelbase, and overall height.

### Folder Structure 

`dataset` folder contains the engineering.csv dataset used in the project.

`Custom_Bike_Pricing_Optimizer_with_pipeline.ipynb` notebook that was used to develop and train the model.

`app.py` is the Flask application file that defines the API endpoints and loads the saved model.

`models` is folder that contain the serialized machine learning models that is used for prediction.

`README.md` is the project documentation file.

`requirements.txt` lists the Python dependencies required to run the project.

`templates` folder contains the HTML templates for the web application.

`static` folder contains the images.


### Prerequisites

To run the project, you must have the following installed on your system:

* Python 3.6+
* Flask
* Pandas
* Scikit-learn

You can install the required packages using the following command:
pip install -r requirements.txt

## Preprocessing
To preprocess the data, we use a `ColumnTransformer` object that applies one-hot encoding to categorical features and passes numerical features through as-is. The resulting transformed data is fed into a `RandomForestRegressor` model that is trained to predict the price of the motorcycle.

## Results
The model achieved a R2 score of 89% and a cross_val_score(cross validation) of 80%, which demonstrates its accuracy in predicting price of bike on custom parameters.

## Deployment
The project was successfully deployed on Render and can be accessed here:
https://custom-bike-pricing-optimizer.onrender.com/

## Conclusion
This project demonstrates the effectiveness of machine learning in predicting the price of custom-built motorcycles. The trained model can be used by motorcycle manufacturing companies to make informed decisions for their next model update.
