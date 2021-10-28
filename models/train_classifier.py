# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# load data from database
def load_data(database_filepath):   
    """
    This function is to load the dataset from the database_filepath.
    It returns
        X: a dataframe of messages
        y : a 36-category dataframe
        category_names : 36 category names
    """
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table(str(database_filepath), engine)
    X = df['message'].values
    y = df.iloc[:, 4:].values
    y_category_names = list(df.iloc[:, 4:].columns)
    return X, y, y_category_names

def tokenize(text):
    """This function is to normalize, remove stop words, stemme and lemmatize the input.
    It return tokenized text """
    token = word_tokenize(text)
    lemma = WordNetLemmatizer()

    lemma_tokens = []
    for x in token:
        lemma_x = lemma.lemmatize(x).lower().strip()
        lemma_tokens.append(lemma_x)

    return lemma_tokens   


def build_model():
    """ This function is to create a pipeline which is later used for training model"""
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {  'vect__ngram_range': ((1, 1), (1, 2)),
                'vect__max_df': (0.5, 0.8),
                'vect__max_features': (None, 50, 100),
                'tfidf__use_idf': (True, False)}

    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """This function is to print out the accuracy, precision, recall scores of 36 categories"""
    y_predict = model.predict(X_test)
    for i in range(len(category_names)):
        print(i, '. ', category_names[i], '. \t acc = ', (y_predict[:, i] == Y_test[:,i]).mean())
        print(classification_report(Y_test[:,i], y_predict[:,i]))


def save_model(model, model_filepath):
    """ Save model with the best estimator"""
    pickle.dump(model, open(model_filepath, 'wb'))


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
