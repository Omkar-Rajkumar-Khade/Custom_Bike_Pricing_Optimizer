# Custom Bike Pricing Optimizer using Machine Learning

This project aims to provide a machine learning solution for predicting the price of custom-built motorcycles based on their various features such as bike name, brand, max power, max torque, fuel tank capacity, top speed, front brake type, kerb weight, overall length, overall width, wheelbase, and overall height. It aims to make decision making of motercycle manufacturing companies easier for next model update. It utilizes several regression algorithms and data preprocessing techniques to optimize the accuracy of the predictions.

## Installation 

1. Clone the repository from GitHub:
`git clone https://github.com/Omkar-Rajkumar-Khade/Custom_Bike_Pricing_Optimizer.git`

2. Install the required dependencies using pip:
`pip install -r requirements.txt`


## Getting Started

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
* pickle
* Scikit-learn

You can install the required packages using the following command:
pip install -r requirements.txt

## How to Use
To get started with this project, follow these steps:
* Clone this repository
* Install the required dependencies using pip: pip install -r requirements.txt
* Start the Flask server: python app.py
* Send a POST request to the /predict endpoint with a JSON payload containing the values for TLR score, RPC score, GO score, OI score, Perception score, and Peer * Perception score.

## Results
The model achieved a R2 score of 89% and a cross_val_score(cross validation) of 80%, which demonstrates its accuracy in predicting price of bike on custom parameters.

## Deployment
-- 
