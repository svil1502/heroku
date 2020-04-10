from flask import Flask, render_template, request   
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
MAX_FILE_SIZE = 1024 * 1024 + 1

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    def custom_rating(genre):
        if (genre == 1 or genre == 2 or genre == 12) :
            return 1
        elif (genre == 3 or genre == 4 or genre == 5):
            return 2
        elif (genre == 6 or genre == 7 or genre == 8):
            return 3
        elif (genre == 9 or genre == 10 or genre == 11):
            return 4


    def prepare_data(data):
        data.Sum= data.Sum.replace(r'\s+','',regex=True)
        data.Sum = data.Sum .str.replace(',', ".").astype(float)
        data.y  = data.y .replace(r'\s+','',regex=True)
        data.y  = data.y .astype(float)
        data.Con= data.Con.replace(r'\s+','',regex=True)
        data.Con = data.Con.str.replace(',', ".").astype(float)
        data.index = pd.to_datetime(data['Time'])
        data = data.resample('D').asfreq().fillna(0)
        data.Time = data.index 
        data = (data.groupby(pd.Grouper(key="Time", freq="W-MON"))
                        [["Con", "Sum", "y"]]
                        .sum()
                        .reset_index())
        #data.Time = data.index
        #data["weekday"] = data['Time'].dt.weekday
        #data['is_weekend'] = data.weekday.isin([5,6])*1
        data['month'] = data['Time'].dt.month
        data['week'] = data['Time'].dt.week
        data['season'] = data['month']
        data['season'] = data.apply(lambda x: custom_rating(x['season']), axis = 1)

        data['lag_con'] = data.Con
        data['lag_sum'] = data.Sum

        data['mov_avg_Con'] = data['Con'].rolling(window=(7,10), min_periods=1, win_type='gaussian').mean(std=0.1)
        data['mov_avg_Sum'] = data['Sum'].rolling(window=(7,10), min_periods=1, win_type='gaussian').mean(std=0.1)

        data['month_average_Con'] = data['Con'].rolling(window=(30,10), min_periods=1, win_type='gaussian').mean(std=0.1)
        data['month_average_Sum'] = data['Sum'].rolling(window=(30,10), min_periods=1, win_type='gaussian').mean(std=0.1)

        data['tree_month_average_Con'] = data['Con'].rolling(window=(90,10), min_periods=1, win_type='gaussian').mean(std=0.1)
        data['tree_month_average_Sum'] = data['Sum'].rolling(window=(90,10), min_periods=1, win_type='gaussian').mean(std=0.1)
        lag_start=1
        lag_end=8
        test_size=0.1

        test_index = int(len(data.y)*(1-test_size))
        for i in range(lag_start, lag_end):

            
            data["lag_con{}".format(i)] = data.Con.shift(i)
            data["lag_sum{}".format(i)] = data.Sum.shift(i)

            data["mov_avg_Con{}".format(i)] =  data.mov_avg_Con.shift(i)
            data["mov_avg_Sum{}".format(i)] =  data.mov_avg_Sum.shift(i)

            data["month_average_Con{}".format(i)] =  data.month_average_Con.shift(i)
            data["month_average_Sum{}".format(i)] =  data.month_average_Sum.shift(i)

            data["tree_month_average_Con{}".format(i)] =  data.tree_month_average_Con.shift(i)
            data["tree_month_average_Sum{}".format(i)] =  data.tree_month_average_Sum.shift(i)
            
        data = data.dropna()
        data = data.reset_index(drop=True)
        X_train = data.loc[:test_index].drop(["y", "Con", "Sum"], axis=1)
        y_train = data.loc[:test_index][["y"]]
        X_test = data.loc[test_index+1:].drop(["y", "Con", "Sum"], axis=1)
        y_test = data.loc[test_index+1:][["y"]]
        return (X_train, y_train, X_test, y_test, data)   
    def func_feature_week(X_predict):
        X_X_predict = pd.DataFrame()
        lag_start = 1
        lag_end = 8
        X_X_predict["Time"]  = X_predict["Time"]
        X_X_predict["month"]  = X_predict["month"]
        X_X_predict["week"]  = X_predict["week"]
        X_X_predict["season"]  = X_predict["season"]
        for feat in feature:
            X_X_predict["{feat}".format(feat=feat)]  =data["{feat}".format(feat=feat)].rolling(7,min_periods=1).mean()
            for i in range(lag_start, lag_end):
                    X_X_predict["{feat}{i}".format(feat=feat, i=i)]  = X_predict["{feat}{i}".format(feat=feat, i=i)] 
        return X_X_predict
    def prediction(dataset):
        dataset.columns = ["Time","Con","y", "Sum"]
        data = dataset.copy()
        (X_train, y_train, X_test, y_test, data) = prepare_data(data)
        #Проверим 119 строку y_test-должно быть 1235
        X_predict = X_test.loc[119:119]
        feature = ["lag_con","lag_sum","mov_avg_Con","mov_avg_Sum","month_average_Con","month_average_Sum", "tree_month_average_Con", "tree_month_average_Sum"]
        #Функция для подготовки данных для датасета, которых нужно спрогнозировать
        X_X_predict = func_feature_week(X_predict)
        new_df = X_X_predict[['Time', 'month', 'week', 'season', 'lag_con', 'lag_sum', 'mov_avg_Con',
                    'mov_avg_Sum', 'month_average_Con', 'month_average_Sum',
                    'tree_month_average_Con', 'tree_month_average_Sum', 'lag_con1',
                    'lag_sum1', 'mov_avg_Con1', 'mov_avg_Sum1', 'month_average_Con1',
                    'month_average_Sum1', 'tree_month_average_Con1',
                    'tree_month_average_Sum1', 'lag_con2', 'lag_sum2', 'mov_avg_Con2',
                    'mov_avg_Sum2', 'month_average_Con2', 'month_average_Sum2',
                    'tree_month_average_Con2', 'tree_month_average_Sum2', 'lag_con3',
                    'lag_sum3', 'mov_avg_Con3', 'mov_avg_Sum3', 'month_average_Con3',
                    'month_average_Sum3', 'tree_month_average_Con3',
                    'tree_month_average_Sum3', 'lag_con4', 'lag_sum4', 'mov_avg_Con4',
                    'mov_avg_Sum4', 'month_average_Con4', 'month_average_Sum4',
                    'tree_month_average_Con4', 'tree_month_average_Sum4', 'lag_con5',
                    'lag_sum5', 'mov_avg_Con5', 'mov_avg_Sum5', 'month_average_Con5',
                    'month_average_Sum5', 'tree_month_average_Con5',
                    'tree_month_average_Sum5', 'lag_con6', 'lag_sum6', 'mov_avg_Con6',
                    'mov_avg_Sum6', 'month_average_Con6', 'month_average_Sum6',
                    'tree_month_average_Con6', 'tree_month_average_Sum6', 'lag_con7',
                    'lag_sum7', 'mov_avg_Con7', 'mov_avg_Sum7', 'month_average_Con7',
                    'month_average_Sum7', 'tree_month_average_Con7',
                    'tree_month_average_Sum7']]
        X_train['Time']=X_train['Time'].apply(lambda x: x.toordinal())
        X_test['Time']=X_test['Time'].apply(lambda x: x.toordinal()) 
        #Модель 1 Линейная регрессия
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        #На тестовых данных получаем
        t = lr.predict(X_test.loc[119:119])[0][0]
        return t

    args = {"method": "GET"}
    if request.method == "POST":
        t = 0
        file = request.files["file"]
        if bool(file.filename):
            file_bytes = file.read(MAX_FILE_SIZE)
            args["file_size_error"] = len(file_bytes) == MAX_FILE_SIZE
            args["method"] = "POST"   
        try:
                dataset = pd.read_csv(file.filename)
                t=0
        except IOError as e:
                t = file.filename
                
        else:
                t = 1
        args["t"]  = t
        
    return render_template("main.html", args=args)


if __name__ == "__main__":
    app.run(debug=True)    
