from flask import Flask, render_template ,request
import numpy as np
import pickle

model = pickle.load(open('mdl.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def rt():
    return render_template('home.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    data1 = 0
    data2 = request.form['a']
    data3 = request.form['b']
    data4 = request.form['c']
    arr = np.array([[data1,data2,data3,data4]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)
if __name__ == '__main__':
    app.run(port =8000,debug=True)