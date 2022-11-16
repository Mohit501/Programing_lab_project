from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/About')
def About():
    return render_template("home.html")


@app.route('/DataAnalysis')
def DataAnalysis():
    return render_template("dataanalysis.html")


@app.route('/ModelDevelopment')
def ModelDevelopment():
    return render_template("model_development.html")


@app.route('/PredictiveModel')
def PredictiveModel():
    return render_template("predictive_model.html")

@app.route('/data_input',methods = ['POST','GET'])
def data_input():
    if request.method == 'POST':
        Name = request.form['Name']
        Gender = request.form['Gender']
        Age = request.form['Age']
        Hypertension = request.form['Hypertension']
        HeartDisease = request.form['HeartDisease']
        Marriage = request.form['Married']
        Work_type = request.form['WorkType']
        Residence = request.form['Residence']
        AverageGlucoseLevel = request.form['AverageGlucoseLevel']
        BMI = request.form['BMI']
        prediction = machine_learning_pipeline(Gender,Age,Hypertension,HeartDisease,Marriage,Work_type,Residence,AverageGlucoseLevel,BMI)

        prediction = {'Prediction': prediction}

        data = {"Name":Name,"Gender":Gender,"Age":Age,"Hypertension":Hypertension,"HeartDisease":HeartDisease,"Marriage":Marriage,
        "Work_type":Work_type,"Residence":Residence,"AverageGlucoseLevel":AverageGlucoseLevel,"BMI":BMI}
        return render_template("Results.html",prediction=prediction,data = data)

def machine_learning_pipeline(Gender,Age,Hypertension,HeartDisease,Marriage,Work_type,Residence,AverageGlucoseLevel,BMI):
    Gender = int(Gender)
    Age = float(Age)
    Hypertension = int(Hypertension)
    HeartDisease = int(HeartDisease)
    Marriage = int(Marriage)
    Work_type = int(Work_type)
    Residence = int(Residence)
    AverageGlucoseLevel = float(AverageGlucoseLevel)
    BMI = float(BMI)

    to_predict = [[Gender,Age,Hypertension,HeartDisease,Marriage,Work_type,Residence,AverageGlucoseLevel,BMI]]
    with open('model_rf_pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(to_predict)
    print(prediction)
    return str(prediction)




if __name__ == '__main__':
    app.run(debug=True)
