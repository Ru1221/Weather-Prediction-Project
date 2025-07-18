import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import pickle

df=pd.read_csv("weather (1).csv")

df["RainToday"]=df["RainToday"].map({'No':0,'Yes':1})

df["RainTomorrow"]=df["RainTomorrow"].map({'No':0,'Yes':1})

x=df[["MinTemp","MaxTemp","Rainfall","Evaporation","WindSpeed3pm","Humidity9am","Humidity3pm",
      "Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","RainToday","RISK_MM",
     ]]
y=df["RainTomorrow"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

classifier_model=DecisionTreeClassifier(random_state=42)

classifier_model.fit(x_train,y_train)

pickle.dump(classifier_model,open("weather_classifier_model.pkl","wb"))




