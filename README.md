# Churn prediction-with-clean-code-and-test

**This repository demonstrates a churn prediction modelling with testing and logging**

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [How to run](#run)
5. [ETL & ML Pipelines](#model)
6. [Licensing, Authors, and Acknowledgements](#licensing)

### Installation <a name="installation"></a>
- Python 3.8+
- ML Libraries: NumPy, Pandas, Sciki-Learn
- NLP Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly
- Additional: Bootstrap for styling, HTML for templates


### Project Motivation<a name="motivation"></a>
This projects aims to demonstrate (except monitoring) all steps in a Data Science project.
The data is collected and labeled by Figure Eight(now Appen). The data consist of messages collected via 3 genres;
    - Direct
    - Social Media
    - News
    
The messages labelled into 36 categories by Appen.
Our model aims to take the messages as input and try to predict these labels in order to send or warn authorities.

1. ETL
    - Extract data : Input data from Appen
    - Transform data : clean  and transfor data
    - Load data : Load data to a SQL database
2. ML Pipeline
    - Initializing the data
    - Vectorization and transfromation of the texts
    - Build model
    - Train model
    - Display the results and metrics
    - Save model
3. Deployment 
    - Create a web-app with Flask and Plotly, and using Bootstrap templates.

It is also demonstrating NLP modeling with NLTK and scikit-learn.


### File Descriptions <a name="files"></a>

There are 3 folders each one is related with the above steps;

    - app:
        this folder contains run.py file which is running the web-app
        -templates: there are two templates
    - data:
        this folder contains the ETL pipeline files.
        - Input files: disaster_messages.csv and disaster_categories.csv
        - Clean and transform: process_data.py
        -Load data : Load data into a database DisasterResponse.db
     - model:
        this folder contains modelling part of the files
        - train_classifier.py : the script that runs and trains the model
        - classifier.pkl : Pickle file where the model is saved
        

### How to run <a name="run"></a>

If you want to clone this repository and run the program:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app` and Run your web app: `python run.py`

 



### Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Appen for the data, and templates for Udacity.
