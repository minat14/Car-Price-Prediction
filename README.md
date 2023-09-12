# Car Price Prediction Exercise

Running app.py deploys a pre-trained machine learning model (model.joblib and encoder.joblib as one-hot-encoder) using Flask with main.html (inside templates folder) containing the formatting
The machine learning model used was random forest regressor. The model takes input features about a car (make, model year, transmission type, fuel type, mileage) and returns an estimated price. 

To run app.py create a virtual environment which includes the files app.py, encoder.joblib, model.joblib, and the templates folder holding main.html. Change directory in the terminal to this virtual environment and run python app.py
(You will need to have Flask installed)
