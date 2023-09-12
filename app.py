#import libraries
import flask
from joblib import dump, load
import pandas as pd
import numpy as np
import datetime

#loading pre-trained model and one hot encoder
mod = load('model.joblib')
enc = load('encoder.joblib')

#initialise flask app
app = flask.Flask(__name__, template_folder='templates')

#default page for url ending '/'
@app.route('/', methods=['GET', 'POST'])
def main():
    #returning main.html page with form so it can be filled in to get car details 
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
        
    #returning results from form submission
    if flask.request.method == 'POST':
        #extracting inputs given by form and assigning to variables
        make = flask.request.form['make']
        year = flask.request.form['year']
        fuel = flask.request.form['fuel']
        transmission = flask.request.form['transmission']
        mileage = flask.request.form['mileage']

        #using loaded encoder to one hot encode categorical features
        features = enc.transform([[make, fuel, transmission]]).toarray()
        #getting age of car from model year
        current_year = datetime.datetime.now().year
        year = current_year - int(year)
        #making array of car features to go through model
        X = np.insert(features[0], 0, [current_year - int(year), int(mileage)])
        #getting prediction from loaded model 
        pred = mod.predict([X])[0]

        #returning main.html page with features inputted and results displayed
        return flask.render_template('main.html',
                                    original_input={'Make':make,
                                                    'Age':year,
                                                    'Fuel':fuel,
                                                    'Transmission':transmission,
                                                    'Mileage':mileage},
                                    result=pred)

#run app
if __name__ == "__main__":
    app.run()