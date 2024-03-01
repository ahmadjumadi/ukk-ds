import pandas as pd

from flask import Flask, request, render_template
import joblib

model = joblib.load("model/model.sav")
app = Flask(__name__)


## proses nampilkan form input data
@app.route("/", methods=["GET"])
def index():
  return render_template('index.html')

## proses load data
@app.route('/process', methods=["POST"])
def result():
  data = [{
    "precipitation": request.form['precipitation'],
    "temp_max": request.form['temp_max'],
    "temp_min": request.form['temp_min'],
    "wind": request.form['wind'],
  }]

  df = pd.DataFrame(data)
  sc = joblib.load("scaller/scaller.joblib")
  df[df.columns] = sc.transform(df[df.columns])
  y_test = model.predict(df)
  x = ""
  if(y_test==1):
    x = 'Tidak Hujan'
  else:
    x = 'Akan Hujan'

  return render_template('index.html',
                         result = x,
                         precipitation=data[0]['precipitation'],
                         temp_max=data[0]['temp_max'],
                         temp_min=data[0]['temp_min'],
                         wind=data[0]['wind'])
