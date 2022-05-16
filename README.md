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

3. Go to http://0.0.0.0:3000/

- Nav panel & Message query:
![image](https://user-images.githubusercontent.com/85060022/157391053-0a674e54-0270-4c6c-aebe-1a15a580a980.png)

You can use navigation panel for refreshing page after query, go to Udacity or this repository.

You can enter a message to classify. You can see the message as an example and how the model predicted labels of this messages.
![image](https://user-images.githubusercontent.com/85060022/157391260-b1071b07-5e56-4405-a3b0-d6c8074fec26.png)

![image](https://user-images.githubusercontent.com/85060022/157391330-726853e1-3523-4145-b05c-7ab9ded78dd1.png)


- Graphs:
    The first graph shows the distributions of the sources.
![image](https://user-images.githubusercontent.com/85060022/157391758-ede21116-9cb8-437a-9f79-1c30add6f5e1.png)

    Second graph  shows the distribution of the labels. How these messages labelled by Appen.
![image](https://user-images.githubusercontent.com/85060022/157391899-b6332cc7-76de-4e9b-9c78-8a60a76895d6.png)



### ETL & ML Pipelines <a name="model"></a>
**ETL**
There are 2 csv file one is for "messages" collected during disasters, the other one is the "categories" that messages belong to, labeled by Appen.

- Read csv files into dataframe.
- Merge these 2 dataframes
- Clean and process data for ML algorithms
- Load data into a database using SQLalchemy.

**ML**
- Load data from database.
- Tokenize data with NLTK library

    **Model**
       
       - Build a pipeline:
            - Vectorize and transform using sklearn's CountVectorizer() and TfidfTransformer() functions
            - Uses Multioutput classifier, and estimator is RandomForestClassifier
            - Make a GridSearch for optimizing parameters.
       - Evaluate the model
            
            Using Classification report
            
       - Save model into a pickle file.
 The data is unbalanced so it is hard to get  good results for some categories. 
 
 As an example there is a category for child_alone, but no message is labelled into this category.
 
 The aim of this project is to demonstrate an end-to-end Data Science process, so there is a way to go on data cleaning and modelling.
    - Get much more data for all labels
    - Use pre-trained deep learning models ..etc
 



### Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Appen for the data, and templates for Udacity.
