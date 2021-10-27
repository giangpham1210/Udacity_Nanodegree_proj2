# Udacity_Nanodegree_project 2: Disaster Response Pipeline Project

### Project overview
This project is to build a model for an API. It can classify disaster messages which are really sent during disaster events. An ML pipeline is created to categozied these events, then, the messages can be sent to the appropriate agency.

There are 3 main parts in the project, which are an ETL pipeline, an ML pipeline and a webapp.

### Installation:
The following Python libraries are used in this project: NumPy, Pandas, Sys, Matplotlib, Plotly, Nltk, Sklearn, Sqlalchemy, Pickle, Seaborn, Json, Flask.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/Classifier.pkl`

2. Run the following command in the app's directory to run your web app.
        `python run.py`

3. Go to http://0.0.0.0:3001/

### Data and Code:
    - app
    | - templates
    | |- go.html  # webapp page
    | |- master.html  # webapp main page
    |- run.py  # Flask file

    - data
    |- DisasterResponse.db   # database to save clean data to
    |- disaster_categories.csv  # data to process 
    |- disaster_messages.csv  # data to process
    |- process_data.py

    - models
    |- Classifier.pkl  # printed model 
    |- train_classifier.py

    - README.md
    
 ### SQLite database - Visualization
 ![alt text](https://github.com/giangpham1210/Udacity_Nanodegree_proj2/blob/main/prj2_plot.PNG)
 
 ![alt text](https://github.com/giangpham1210/Udacity_Nanodegree_proj2/blob/main/prj2_plot2.PNG)
