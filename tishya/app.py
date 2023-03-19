from flask import Flask, render_template ,request, url_for,redirect
import numpy as np
import pickle

model1 = pickle.load(open('./model1.pkl','rb'))
model2 = pickle.load(open('./model2.pkl','rb'))
model3 = pickle.load(open('./model3.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def rt():
    return render_template('home.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('rt'))
    else:
        age = float(request.form['age'])
        edu = float(request.form['edu'])
        cigs = float(request.form['cigs'])
        bpmeds = float(request.form['bpmeds'])
        prestr = float(request.form['prestr'])
        prehyp = float(request.form['prehyp'])
        chol = float(request.form['chol'])
        sysbp = float(request.form['sysbp'])
        diabp = float(request.form['diabp'])
        bmi = float(request.form['bmi'])
        hr = float(request.form['hr'])
        glc = float(request.form['glc'])
        gender = float(request.form['gender'])
        cursm = float(request.form['cursm'])
        diab = float(request.form['diab'])
        arr = np.array([[age,edu,cigs,bpmeds,prestr,prehyp,chol,sysbp,diabp,bmi,hr,glc,gender,cursm,diab]])
        pred1 = model1.predict(arr)
        pred2 = model2.predict(arr)
        pred3 = model3.predict(arr)
        return render_template('predictions.html', pred1=pred1, pred2=pred2, pred3=pred3,l=[age,edu,cigs,bpmeds,prestr,prehyp,chol,sysbp,diabp,bmi,hr,glc,gender,cursm,diab],zip=zip)
    
if __name__ == '__main__':
    app.run(port=5000,debug=True)