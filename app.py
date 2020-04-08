import flask
from flask import render_template
import pickle
import sklearn
# import numpy package for arrays and stuff 
import numpy as np  
  
# import matplotlib.pyplot for plotting our result 
import matplotlib.pyplot as plt 
  
# import pandas for importing csv files  
import pandas as pd  
import sys
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import xgboost as xgb

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html' )
        
    if flask.request.method == 'POST':
        temp = 1
        with open('model.pkl', 'rb') as fh:
            loaded_model = pickle.load(fh)
        exp = float(flask.request.form['experience'])
        #temp = loaded_model.predict([ [exp] ])
        temp = 4
        return render_template('main.html', result = temp)

if __name__ == '__main__':
    app.run()