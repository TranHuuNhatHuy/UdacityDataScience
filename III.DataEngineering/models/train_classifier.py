"""
Classifier Trainer for Disaster Response Pipeline (Udacity - Data Scientist Nanodegree)

Syntax
--------
    python train_classifier.py <db_filepath> <pickle_filepath>

Example
----------
    python train_classifier.py DisasterResDB.db best_model.pkl

Parameters
--------------
    db_filepath (str) :
        SQLite database filepath
    pickle_filepath (str) :
        Pickle file name for saving the ML model
"""

# import libraries
import sys
import pandas as pd 
import re
import pickle
from sqlalchemy import create_engine
from scipy.stats import gmean
from matplotlib import pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    """
    Load and preprocess data from the SQLite database.

    Parameters
    --------------
        database_filepath (str):
            Path to the SQLite destination database.

    Returns
    ----------
        X (pandas Series): 
            Messages data for training.
        Y (pandas DataFrame): 
            Categories data for training.
        categories (Index): 
            Category names.
    """
    # load data from database
    engine = create_engine('sqlite:///DisasterResDB.db')
    df = pd.read_sql_table("DisasterResTable", engine)
    
    # Further cleaning as introduced in the Python Notebook
    df = df.drop("child_alone", axis = 1)
    df.loc[
        df["related"] == 2,
        "related"
    ] = 1
    
    # Extract data for training
    X = df["message"]
    Y = df.iloc[ : , 4 : ]
    categories = Y.columns
    
    return X, Y, categories
    

def tokenize(text):
    """
    Tokenize the texts based on NLP procedures.
    
    Parameters
    --------------
        text (str) : 
            input text messages
    
    Returns
    ----------
        clean_tokens (list) :
            list of tokens extracted from input text messages.
    """
    
    # Replaces URLs with placeholders
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    for url in re.findall(url_regex, text):
        text = text.replace(url, "url_placeholder")
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lemmy, lower, strip them (this sounds weird to say.... God forgive me)
    lemmy = WordNetLemmatizer()
    clean_tokens = []
    
    for token in tokens:
        clean_tokens.append(lemmy.lemmatize(token).lower().strip())
        
    return clean_tokens


def build_model(model = None):
    """
    Build a machine learning model pipeline.
    
    Parameters
    --------------
        model :
            an input model. None by default. If None, a GridSearchCV model
            will be generated.

    Returns
    ---------
        pipeline (sklearn.pipeline.Pipeline): 
            Machine learning model pipeline.
    """
    if model != None:
        return model
    
    pipeline = Pipeline([(
        "features", FeatureUnion([(
            "text_pipeline", 
            Pipeline([(
                "count_vectorizer", 
                CountVectorizer(tokenizer = tokenize)
            ),(
                "tfidf_transformer", 
                TfidfTransformer()
            )]))])
        ),(
            "classifier",
            MultiOutputClassifier(AdaBoostClassifier())
    )])
    
    parameters = {
        "classifier__estimator__n_estimators" : [10, 50],
        "classifier__estimator__learning_rate" : [0.5, 0.1]
    }
    
    cv = GridSearchCV(
        pipeline,
        param_grid = parameters,
        scoring = "f1_micro",
        cv = 2,
        n_jobs = -1,             # All processors go boom!
        verbose = True
    )
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the machine learning model.

    Parameters
    --------------
        model :
            Trained machine learning model.
        X_test (pandas Series) :
            Test data.
        Y_test (pandas DataFrame) :
            True labels.
        category_names (list) :
            Category names.
    """

    pred_Y_test = model.predict(X_test)

    pred_report = classification_report(
        Y_test.values,
        pred_Y_test,
        target_names = category_names
    )
    
    print(pred_report)


def save_model(model, model_filepath):
    """
    Save the trained machine learning model to a pickle file.

    Parameters
    --------------
        model:
            Trained machine learning model.
        model_filepath (str):
            Path to the pickle file.
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


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