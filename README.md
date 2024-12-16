# Visa Prediction Classification Using Machine Learning.

## Project Overview:

This project aims to predict the outcome of visa applications using machine learning. The system takes a number of inputs from the user about their profile and makes a prediction, whether that person would get approval or rejection on their US visa application.

https://github.com/user-attachments/assets/bc19911f-7e95-40e6-b839-c30c4cc9e9c3


## Methods Used:

- Inferential Statistics
- Machine Learning
- Data Visualization
- Predictive Modeling


## Technologies Used: 
- VS code, Windows
- Python, Anaconda, Git, Jupyter   
- Sklearn, Pandas, Matplotlib, Seaborn, Flask

### Project Description:
Kaggle.com provides a dataset of 1500 instances, with a number of numerical and categorical features. We use Pandas and seaborn to visualise the data distribution like 'prevailing_wage' and 'no_of_employees_of_organisation'. Then, an array of classifier models (xgboost, SVC,Adaboost, Random Forest etc.). Model is evaluated based on recall score so as to improve true positive case.For Adaboost classifier, it is 0.90 and hence it is choice for training the model. 

![visa_pred_datasset](https://github.com/user-attachments/assets/36f79f48-0fd4-488b-9400-81b556c9e698)

![visa_approval_dist](https://github.com/user-attachments/assets/2b4d8782-eadb-4505-9bb7-2bc29e65e077)


## Getting Started 

1. Clone the repository into your VS code inside a folder.
```
git clone https://github.com/Rgarlay/Visa_Prediction.git
```
2. Activate the virtual environment and install all dependencies
```
conda activate venv/
```
pip install -r path_to_txt_file/req.txt 
```
3. Run the code to train the model
NOTE: Change all pre defined local paths to the paths in your local environment.
```
py -m src.components.data_ingestion.py
```
4. Run the app.py to run the model locally

```
py app.py
```


### Contact:
For Any Further Query, you can contact me at : ravigarlay@gmail.com





