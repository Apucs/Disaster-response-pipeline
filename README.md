# Disaster Response Pipeline Project

## __Introduction:__
    In the Project, I have worked with a data set containing real messages that were sent during disaster events. 
    I have created a machine learning pipeline to categorize these events so that people can send the messages to an 
    appropriate disaster relief agency.
    
    This project includes a web app built with flusk and woth plotly visualization where an emergency worker can input
    a new message and get classification esults in several categories. This web display visualizations of the data.
    
    

### __Instructions:__
> - __Install the dependencies:__
    `pip install -r requirements.txt `
> - __Clone the repository:__ `git clone https://github.com/Apucs/Disaster-response-pipeline.git ` 

### __Details:__
> This project has two notebooks where every steps have been shown clearly and distinguishly.
1. ```ETL Pipeline Preparation.ipynb``` : In this notebook I have merged the dataset in the data folder 
    (disaster_messages.csv and disaster_categories.csv), then do some preprocessing to get the clean data. 
    I stored the clean data in sqlite database DisasterResponse.db which is also in the data folder. Finally,
    `process_data.py` in the data folder is created from this notebook to implement the web app.

2. ```ML Pipeline Preparation.ipynb``` : In this notebook, I have loaded data from the database created in the 
    earlier stage and prepared a machine learning pipeline and Trained and tuned the model using GridSearchCV. 
    From this notebook later I have managed the `train_classifier.py` in the models folder.

### __Train the model:__
To train the model-
1. Run the following commands in the project's root directory to set up database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

### __Use the pretrained model__
To use the pretrained model: 
    - Pretained model can be downloaded from [here](https://drive.google.com/file/d/1LMp5KkW5fd-rB_VAlSVc16H9UquxuR4r/view?usp=sharing)
    - Place that model to ```model``` directory
    - Then run `python run.py`
    
    
    
    
    
    

LICENSE: This project is licensed under the terms of the MIT license.
