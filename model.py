import pickle

classifier_model=pickle.load(open("weather_classifier_model.pkl","rb"))

y_pred = classifier_model.predict([[8.0,24.3,0.0,3.4,20,68,29,1019.7,1015.0,7,7,14.4,23.6,0,3.6]])
print(y_pred)
print(y_pred[0])


