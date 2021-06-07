# Disaster Response Pipeline Project

### Background 
...

### Repository Contents
```bash
.
├── README.md
├── app                                 # Disaster response Flask application files
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data                                # Raw data, etl script, and sqlite database
│   ├── categories.csv
│   ├── disaster_response.db
│   ├── messages.csv
│   └── process_data.py
├── models                              # Model training script, performance results, and picked classifier
│   ├── classifier.pkl
│   ├── f1_score_performance.png
│   ├── model_performance.csv
│   └── train_classifier.py
├── notebooks                           # Experimentation notebooks
│   ├── ETL Pipeline Preparation.ipynb
│   ├── ML Pipeline Preparation.ipynb
└── requirements.txt


```



### Project Set-up:
**Python Version:** 3.8.1  
Create a clean virtual environment and install the package requirements:   
```bash 
pip install -r requirements.txt 
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
