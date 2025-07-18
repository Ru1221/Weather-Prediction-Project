from flask import Flask,request,render_template
import pickle

classifier_model=pickle.load(open("weather_classifier_model.pkl","rb"))

app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def predition():
    if request.method == "GET":
        return render_template("weatherprediction.html")
    elif request.method == "POST":
        min_temp=float(request.form["MinTemp"])
        max_temp = float(request.form["MaxTemp"])
        rf= float(request.form["Rainfall"])
        e = float(request.form["Evaporation"])
        w3pm = float(request.form["WindSpeed3pm"])
        h9am = float(request.form["Humidity9am"])
        h3pm = float(request.form["Humidity3pm"])
        p9am = float(request.form["Pressure9am"])
        p3pm= float(request.form["Pressure3pm"])
        c9am = float(request.form["Cloud9am"])
        c3pm = float(request.form["Cloud3pm"])
        t9am = float(request.form["Temp9am"])
        t3pm = float(request.form["Temp3pm"])
        rt = float(request.form["RainToday"])
        rm = float(request.form["RiskMM"])
        y_pred = classifier_model.predict([[min_temp,max_temp,rf,e,w3pm,h9am,h3pm,p9am,p3pm,c9am,c3pm,t9am,t3pm,rt,rm]])
        if y_pred[0] == 0:
            return render_template("notraintomorrow.html")
        elif y_pred[0] == 1:
            return render_template("raintomorrow.html")
if __name__=="__main__":
    app.run(debug=True)

