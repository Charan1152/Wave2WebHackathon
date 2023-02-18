#linear_regression_
import pandas as pd 
import numpy as np
import sklearn as sk
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv(r"data.csv")
data = data.drop(["FLOW_DATE"],axis=1)

X =data.drop(["PRESENT_STORAGE_TMC"],axis=1)

Y =data['PRESENT_STORAGE_TMC']
Y = Y.values.reshape(-1,1)

day_index = 499
days =[i for i in range(Y.size)]

clf = LinearRegression()
s = clf.fit(X,Y)

import pickle
pickle.dump(s,open('mdl.pkl','wb'))



plt.scatter(days,Y,color='g')
plt.scatter(days[day_index],Y[day_index],color='g')
plt.title('Storage level')
plt.xlabel('days')
plt.ylabel('Storage level in tmc')

sns.pairplot(data[["PRESENT_STORAGE_TMC","RES_LEVEL_FT","INFLOW_CUSECS","OUTFLOW_CUECS"]],diag_kind='kde')
