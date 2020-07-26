import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pickle
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

def load_data(database_filepath):
    """Load data from the database filepath
    & store them in X and y variable
    args:
        database_filepath: path to the database file
    returns:
        X, y & the list of categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X,Y,category_names


def tokenize(text):
    """
    args:
        Input value
    returns:
        returns the tokenize version of the input
    """
    
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect and replace urls
    detected_urls = re.findall(url_re, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize sentences
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # save cleaned tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    # remove stopwords
    STOPWORDS = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]
    
    return clean_tokens


def build_model():
    """
      build NLP pipeline - count words, tf-idf, multiple output classifier,
      grid search the best parameters
    Args: 
        None
    Returns: 
        cross validated classifier object
    """   
    pipeline = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100, n_jobs = -1)))
    ])
    
    parameters = {
    'clf__estimator__n_estimators':[50,100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs= -1)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluate the model performances, in terms of f1-score, precison and recall
    Args: 
        model: the model to be evaluated
        X_test: X_test dataframe
        Y_test: Y_test dataframe
        category_names: category names list defined in load data
    Returns: 
        perfomances (DataFrame)
    """   
    Y_pred = model.predict(X_test)
    
    # build classification report on every column
    performances = []
    for i in range(len(category_names)):
        performances.append([f1_score(Y_test.iloc[:, i].values, Y_pred[:, i], average='micro'),
                             precision_score(Y_test.iloc[:, i].values, Y_pred[:, i], average='micro'),
                             recall_score(Y_test.iloc[:, i].values, Y_pred[:, i], average='micro')])
    # build dataframe
    performances = pd.DataFrame(performances, columns=['f1 score', 'precision', 'recall'],index = Y_test.columns)   
    
    return performances


def save_model(model, model_filepath):
    """
        Save model to pickle
    """
    pickle.dump(model, open('model_filepath', 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()