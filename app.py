from flask import Flask,request
from flask.templating import render_template
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


app = Flask(__name__) 


@app.route('/first')
def welcome():
    return "The first line of code here"

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template("home.html")
    else:
        data=CustomData(
                    continent = request.form.get('continent'),
                    education_of_employee = request.form.get('education_of_employee'),
                    has_job_experience = request.form.get('has_job_experience'),
                    requires_job_training = request.form.get('requires_job_training'),
                    no_of_employees = float(request.form.get('no_of_employees')),
                    yr_of_estab = float(request.form.get('yr_of_estab')),
                    region_of_employment = request.form.get('region_of_employment'),
                    prevailing_wage = float(request.form.get('prevailing_wage')),
                    unit_of_wage = request.form.get('unit_of_wage'),
                    full_time_position = request.form.get('full_time_position')

)
        pred_df=data.get_data_into_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template("home.html",results=results[0])



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)