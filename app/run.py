import json
import plotly
import pandas as pd
import sys
import os
sys.path.append("./models")
from train_classifier import GloveVectorizer, tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///./data/disaster_response.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load('./models/classifier.pkl')

# Need to instantiate the GloveVectorizer for pickled pipeline to work.
GloveVectorizer()


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    model_performance = pd.read_csv("./models/model_performance.csv")
    model_performance = model_performance.sort_values("f1_score", ascending=False)
    f1_categories = model_performance['category']
    f1_value = model_performance["f1_score"]

#    categories = df.iloc[:, 4:].melt(var_name="categories", value_name="count")
#    categories = categories.groupby("categories")['count'].sum().reset_index()
#    categories = categories.merge(model_performance, left_on="categories", right_on="category")
#    categories = categories.sort_values("f1_score", ascending=False)
#    category_values = categories["categories"]
#    category_count = categories["count"]

    categories = df.iloc[:, 3:].melt(id_vars="genre", var_name="categories", value_name="count")
    categories = categories.groupby(["genre", "categories"])['count'].sum().reset_index()
    category_count = categories.groupby("categories")['count'].sum().reset_index()
    category_count = category_count.rename(columns={"count": "category_count"})
    categories = categories.merge(category_count, on="categories")
    categories = categories.sort_values("category_count", ascending = False)
    data_obj = []
    for genre in genre_names:
        category_values = categories.loc[categories["genre"] == genre, "categories"]
        category_count = categories.loc[categories["genre"] == genre, "count"]
        data_obj.append(Bar(x=category_values,
                            y=category_count,
                            name=genre))




    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
            {'data': data_obj,
            'layout': {
                'title': '# of Messages per Category',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                },
            'margin': {'b': 120
                }
            }
        },
            {'data': [
                Bar(
                    x=f1_categories,
                    y=f1_value
                )
            ],
            'layout': {
                'title': 'F1-Score on Test Set',
                'yaxis': {
                    'title': "F1-Score"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                },
            'margin': {'b': 120}
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()