import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import cgi, csv
import datetime
from datetime import timedelta
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier

from flask import Flask, render_template, request   

#import matplotlib.pyplot as plt
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
    def data(data):
                data.Sum= data.Sum.replace(r'\s+','',regex=True)
                data.Sum = data.Sum.str.replace(',', ".").astype(float)
                data.y  = data.y.replace(r'\s+','',regex=True)
                data.y = data.y.str.replace(',', ".").astype(float)
                data.Con= data.Con.replace(r'\s+','',regex=True)
                data.Con = data.Con.str.replace(',', ".").astype(float)
                return data
    def data_shot(df):
                df['Sum'] = df['Sum'].str.replace(',', ".").astype(float)
                df['y'] = df['y'].str.replace(',', ".").astype(float)
                df['Con'] = df['Con'].str.replace(',', ".").astype(float)
                return df
    def prepare_data(data):

                data.index = pd.to_datetime(data['Time'])
                data = data.resample('D').asfreq().fillna(0)
                data.Time = data.index
                data = (data.groupby(pd.Grouper(key="Time", freq="W-MON"))
                                [["Con", "Sum", "y"]]
                                .sum()
                                .reset_index())
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

    def func_feature_week(data):
            feature = ["lag_con","lag_sum","mov_avg_Con","mov_avg_Sum","month_average_Con","month_average_Sum", "tree_month_average_Con", "tree_month_average_Sum"]

            X_X_predict = data.tail(1).copy()

            last =  X_X_predict.copy()
            last.loc[last.index[0],'Time'] =last.iloc[0]['Time'] + timedelta(days=7)
            last.loc[last.index[0],'month']  = last.loc[last.index[0],'Time'].month
            last.loc[last.index[0],'week'] = last.loc[last.index[0],'Time'].week
            last['season'] = last['month']
            last['season'] = last.apply(lambda x: custom_rating(x['season']), axis = 1)
            X_X_predict = last.copy()
            end = data.tail(1).index[0]

            lag_start = 1
            lag_end = 8
            for feat in feature:
                X_X_predict["{feat}".format(feat=feat)]  =data["{feat}".format(feat=feat)].rolling(7,min_periods=1).mean()
                for i in range(lag_start, lag_end):
                    X_X_predict["{feat}{i}".format(feat=feat, i=i)]  = data.loc[data.index[end - (i - 1)],"{feat}".format(feat=feat)]
            return X_X_predict
     #Функция стандартизации данных
    def standart(X_train, X_test, new_df):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_predict_scaled = scaler.transform(new_df)
            X_test_scaled = scaler.transform(X_test)
            return (X_train_scaled, X_test_scaled, X_predict_scaled)
    def prediction_scaler(dataset):
            #dataset.columns = ["Time","Con","y", "Sum"]
            data = dataset.copy()
            (X_train, y_train, X_test, y_test, data) = prepare_data(data)
            feature = ["lag_con","lag_sum","mov_avg_Con","mov_avg_Sum","month_average_Con","month_average_Sum", "tree_month_average_Con", "tree_month_average_Sum"]
            #Функция для подготовки данных для датасета, которых нужно спрогнозировать
            X_X_predict = func_feature_week(data)
            new_df_data = X_X_predict[['Time', 'month', 'week', 'season', 'lag_con', 'lag_sum', 'mov_avg_Con',
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
            new_df = new_df_data.copy()
            X_train['Time']=X_train['Time'].apply(lambda x: x.toordinal())
            new_df['Time']=new_df_data['Time'].apply(lambda x: x.toordinal())
            X_test['Time']=X_test['Time'].apply(lambda x: x.toordinal())
            (X_train_scaled, X_test_scaled, X_predict_scaled) = standart(X_train, X_test, new_df)
            return (X_X_predict, X_train_scaled, X_test_scaled, X_predict_scaled, y_train, y_test, X_train, X_test, data, new_df, new_df_data)
    def proccessing_data(data):
            with open(data, newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                csv_reader = [row for row in csv_reader][1:]
            dataset = pd.DataFrame(csv_reader)
            dataset.columns = ["Time","Con","y", "Sum"]
            return dataset
    #Список всех 6 моделей
    func = [LinearRegression(), DecisionTreeClassifier(random_state=241),KNeighborsClassifier(n_neighbors=5), LogisticRegression(), GaussianNB(), SVC()]
    #Функция обучения на тестовых данных
    def fit_predict_scaler(X_train, y_train, X_test, y_test, lr):
            lr.fit(X_train, y_train)
            prediction = lr.predict(X_test)
            error = mean_absolute_error(prediction, y_test)
            return error
    #Функция моделей с их качеством
    def all_models_data(X_train, y_train, X_test, y_test):
            for lr in func:
                error = fit_predict_scaler(X_train, y_train, X_test, y_test, lr)
                model_all[lr] = error
    #Поиск лучшей модели
    def difference(model_all):
            errors = model_all.values()
            values = model_all.keys()
            errors = list(errors)
            values = list(values)
            temp = errors[0]
            best_model = values[0]
            #best_model[temp] = model_all[temp]
            for er in range(len(errors)):
                if errors[er] < temp:
                    temp = errors[er]
                    best_model = values[er]
            return best_model, temp
    #Функция обучения и прогноза по выбранной модели
    def fit_predict(X_train, y_train, X_predict_scaled,  lr):
            lr.fit(X_train_scaled, y_train)
            prediction = lr.predict(X_predict_scaled)
            return prediction[0]

    args = {"method": "GET"}
    if request.method == "POST":

        file = request.files["file"]
        str_file_value = file.read().decode('utf-8')
        file_t = str_file_value.splitlines()
        csv_reader = csv.reader(file_t, delimiter=',')
        file_data = [row for row in csv_reader][1:]
        dataset = pd.DataFrame(file_data)
        dataset.columns = ["Time","Con","y", "Sum"]
        data = dataset.copy()
        dataset = data_shot(dataset)
        (X_X_predict, X_train_scaled, X_test_scaled,X_predict_scaled, y_train, y_test, X_train, X_test, data, X_predict, X_predict_data)=prediction_scaler(dataset)
        model_all = {}
        all_models_data(X_train_scaled, y_train, X_test_scaled, y_test)
        lr, error = difference(model_all)

        result = fit_predict(X_train_scaled, y_train, X_predict_scaled, lr)
        #(X_X_predict, X_train_scaled, X_test_scaled,X_predict_scaled, y_train, y_test, X_train, X_test, data, X_predict, X_predict_data)=prediction_scaler(data)
        #model_all = {}
        #all_models_data(X_train_scaled, y_train, X_test_scaled, y_test)
        #lr, error = difference(model_all)
        #pr = fit_predict(X_train_scaled, y_train, X_predict_scaled, lr)

        if bool(file.filename):
                            file_bytes = file.read(MAX_FILE_SIZE)
                            args["file_size_error"] = len(file_bytes) == MAX_FILE_SIZE
                            args["method"] = "POST"
                            dataset .to_csv("output.csv")
                            args["t"] = result



        
    return render_template("main.html", args=args)


if __name__ == "__main__":
    app.run(debug=True)    
