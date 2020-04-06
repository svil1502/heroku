import flask
from flask import render_template
import pickle
import sklearn

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
        temp = loaded_model.predict([ [exp] ])
        return render_template('main.html', result = temp)

if __name__ == '__main__':
    app.run()