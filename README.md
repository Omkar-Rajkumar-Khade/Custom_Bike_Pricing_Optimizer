# Custom Bike Pricing Optimizer using Machine Learning

This project aims to provide a machine learning solution for predicting the price of custom-built motorcycles based on their various features such as bike model name, brand, max power, max torque, fuel tank capacity, top speed, front brake type, kerb weight, overall length, overall width, wheelbase, and overall height. It aims to make decision making of motercycle manufacturing companies easier for next model update. It utilizes several regression algorithms and data preprocessing techniques to optimize the accuracy of the predictions.

## Installation

1. Clone the repository from GitHub:

```bash
  git clone https://github.com/Omkar-Rajkumar-Khade/Custom_Bike_Pricing_Optimizer.git
```


2. Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```


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
``` pip install -r requirements.txt ```

## Machine Learning Model Training

To predict the price of a bike, we trained four different regression models:

* Random Forest Regressor (RFR)
* Decision Tree Regressor (DTR)
* Adaptive Boosting Regressor (ABR)
* Elastic Net Regressor (ENR)

The transformers used in the pipeline are:

* `ColumnTransformer`: Used to apply different transformers to different columns of the data.
* `StandardScaler`: Used to standardize the numerical data.
* `OneHotEncoder`: Used to encode categorical data.

The first step in the process was to preprocess the data using ColumnTransformer to one-hot encode the categorical features and scale the numerical features. We used `StandardScaler()` to standardize the numerical data, and `OneHotEncoder(sparse=False, handle_unknown='ignore')` to one-hot encode the categorical data.

We created two different `ColumnTransformer()` instances, one for each model type. The first column transformer, `trf1_RFR`, is used to transform the data before training the RFR and ENR models, and the second column transformer, `trf1_DTR`, is used to transform the data before training the DTR and ABR models.

We then created the four regression models using the following algorithms:

* RFR: RandomForestRegressor(n_estimators=200, * criterion='squared_error')
* DTR: DecisionTreeRegressor(random_state=42)
* ABR: AdaBoostRegressor()
* ENR: ElasticNet(alpha=0.1, l1_ratio=0.5)

We created four pipelines, one for each regression model, using the two column transformers created earlier and the corresponding regression model.

We trained each pipeline on the training data, made predictions using the test data, and evaluated each model's performance using R-squared score. We also used cross-validation with 15 folds to get a better understanding of how well each model generalizes to unseen data.



## Results
After training and testing the four regression models, we found the following results:

| Model           | R-square | Mean R-squared Cross-validation |
| ----------------- |----------------| ------------------------------------------------------------------ |
| Random Forest Regressor (RFR) | 0.88| 0.82 |
| Decision Tree Regressor (DTR) |0.95 | 0.69|
| Adaptive Boosting Regressor (ABR) |0.93 | 0.74|
| Elastic Net Regressor (ENR) |0.74 |0.71 |

It demonstrates its accuracy in predicting price of bike on custom parameters.

## Model Integration
In addition to the machine learning model, this project also includes a Flask app that allows users to interact with the model by providing custom input parameters and getting a price prediction in response. The Flask app is integrated with the machine learning model through the use of the pickle library, which loads the trained model from a saved file.

The Flask app consists of two HTML templates, home.html and result.html, which provide the user interface for entering input parameters and displaying the predicted price. The home.html template contains a form that allows the user to enter the input parameters, and the result.html template displays the predicted price.

The Flask app is run using the app.run(debug=True) command, which starts the development server and enables debug mode. This allows for easy debugging and testing of the app during development. To use the app, simply navigate to the home page (http://localhost:5000/) and enter the desired input parameters. After submitting the form, the predicted price will be displayed on the results page.

## Deployment
The project was successfully deployed on Render and can be accessed here:
https://custom-bike-pricing-optimizer.onrender.com/

## Conclusion
This project demonstrates the effectiveness of machine learning in predicting the price of custom-built motorcycles. The trained model can be used by motorcycle manufacturing companies to make informed decisions for their next model update.
