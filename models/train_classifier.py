# Import packages
import sys
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
sns.set_theme()
import re
from matplotlib import pyplot as plt
import numpy as np

# NLP
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Scikit-Learn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
import spacy
import joblib


class GloveVectorizer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn Glove transformer derived from :
    https://faculty.ai/blog/glove/

    Attributes
    ----------
    model_name : str
        Specific pre-trained model from spacy to instantiate
    _nlp : English
        A spacy text-processing pipeline based on the model_name provided.

    Methods
    -------
    fit(X, y=None):
        unused. Conforms to scikit-learn api style.

    transform(X):
        Applies Glove transformer to text/word.



    """
    def __init__(self, model_name="en_core_web_md"):
        """ Loads spacy model.
        Args:
            Specific pre-trained model from spacy to instantiate
        Returns:
            None
        """
        self.model_name = model_name
        self._nlp = spacy.load(model_name)

    def fit(self, X, y=None):
        """Unused. Meant to conform to  scikit-learn api syle.
        """
        return self

    def transform(self, X):
        """Applies spacy model to X
        Args:
            X: Text to transform
        Returns:
            Glove word vector representation.
        """
        return np.concatenate([self._nlp(doc).vector.reshape(1, -1) for doc in X])



def load_data(database_filepath):
    """ Loads data from sqlite database and filters out disaster categories in which
        there are no messages.

    Args:
        database_filepath: Database file path within project where data is stored
    Returns:
        X (messages), Y (disaster categories)
    """
    engine = create_engine('sqlite:///./{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM messages;", con=engine)
    X = df["message"]
    cols_to_keep = [col for col in df.columns[4:] if df[col].sum() > 0]
    Y = df[cols_to_keep]

    return X, Y


def tokenize(text):
    """ Cleans text by:
        1. normalizing to lowercase
        2. removing punctuation/non-alphanumeric characters
        3. tokenizing
        4. removing stopwords
        5. lemmatizing

    Args:
        text: String of text

    Returns:
        List of tidy words
    """

    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9]", " ",text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words


def build_model():
    """ Constructs scikit-learn grid search model pipeline object using five fold cross-validation
    Args:
        None
    Returns:
        scikit-learn cross-validated pipeline model object
    """

    tfidf_pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ('reduce_dim', TruncatedSVD())])

    glove_pipeline = Pipeline([
        ("glove", GloveVectorizer()),
        ("reduce_dim", TruncatedSVD())])

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("tfidf_pipeline", tfidf_pipeline),
            ("glove_pipeline", glove_pipeline)
        ])),
        ("model", MultiOutputClassifier(GradientBoostingClassifier(random_state=42)))
    ])

    parameters = {
        'features__tfidf_pipeline__reduce_dim__n_components': [100],
        'features__tfidf_pipeline__reduce_dim__random_state': [1234],
        'features__glove_pipeline__reduce_dim__random_state':[1234],
        'features__glove_pipeline__reduce_dim__n_components': [50],
        'model__estimator__max_depth': [5, 10],
        'model__estimator__n_estimators': [100]
    }

    model_pipeline = GridSearchCV(pipeline,
                                  param_grid=parameters,
                                  verbose = 3,
                                  cv = 2,
                                  scoring = "f1_micro")


    return model_pipeline


def build_test_model():
    """
    Test pipeline used to validate script runs without error. Not used in prod.
    """

    test_pipeline = Pipeline([("vect", CountVectorizer(tokenizer = tokenize, max_features=100)),
                              ("tfidf", TfidfTransformer()),
                              ("model", RandomForestClassifier())])

    return test_pipeline


def evaluate_model(model_pipeline, X_test, Y_test):
    """ Generate classification report and f1-score bar plot using test data.

    Args:
        model: scikit-learn cross-validated pipeline model object
        X_test: message data for testing
        Y_test: disaster category data for testing

    Returns:
         classification_report, f1-score plot
    """
    test_preds = model_pipeline.predict(X_test)
    category_names = Y_test.columns

    performance_report = classification_report(Y_test, test_preds, target_names=category_names)
    f1_scores = []
    for i in range(0, len(category_names)):
        f1_scores.append(round(f1_score(Y_test.iloc[:, i], test_preds[:, i], pos_label=1), 2))

    f1_scores = pd.Series(f1_scores, index=category_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=f1_scores.values,
                y=f1_scores.index,
                order=f1_scores.sort_values(ascending=False).index,
                ax=ax)
    ax.set_title("F1 Score for Each Message Category")
    ax.set_xlabel("F1-Score");

    return performance_report, fig


def save_model(model_pipeline, model_filepath):
    """ Pickle model object
    
        Args:
            model: scikit-learn cross-validated pipeline model object
            model_filepath: Location to store model object
        Returns:
            None
    """
    joblib.dump(model_pipeline.best_estimator_, model_filepath, compress=3)

    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model_pipeline = build_model()
        
        print('Training model...')
        model_pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        perf_report, f1_fig = evaluate_model(model_pipeline, X_test, Y_test)
        print(perf_report)
        f1_fig.savefig("./models/f1_score_performance.png")


        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model_pipeline, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/disaster_response.db classifier.pkl')


if __name__ == '__main__':
    main()