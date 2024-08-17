from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from py2neo import Graph
import plotly.express as px
import plotly
import plotly.utils
import json


app = Flask(__name__)

graph = Graph("neo4j+s://6d1b201e.databases.neo4j.io", auth=("neo4j", "2SNPGYwjgRKoi0W6AuAMpPtteNWlJy_HeqgCS38UzCU"))
def load_and_preprocess_data(file_path):
    
    df = pd.read_csv(file_path)


    df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
    df['Updated'] = pd.to_datetime(df['Updated'], errors='coerce')
    df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
    df.dropna(subset=['Created', 'Updated', 'Resolved'], inplace=True)
    df['days_since_updated'] = (pd.Timestamp.now() - df['Updated']).dt.days
    df['resolution_time'] = (df['Resolved'] - df['Created']).dt.days
    df['days_since_created'] = (pd.Timestamp.now() - df['Created']).dt.days
    
    # Map priority to numeric values
    priority_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
    df['Priority_Num'] = df['Priority'].map(priority_mapping)

    return df


df = load_and_preprocess_data('env/Jira (5).csv')

def identify_bottlenecks(query, df, n_clusters=5, contamination=0.05):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    query_vector = vectorizer.transform([query])
    query_cluster = kmeans.predict(query_vector)[0]

    cluster_issues = df[df['cluster'] == query_cluster].copy()

    features = cluster_issues[['resolution_time', 'days_since_updated']].fillna(0)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    cluster_issues.loc[:, 'is_anomaly'] = iso_forest.fit_predict(features)

    bottlenecks = cluster_issues[cluster_issues['is_anomaly'] == -1]
    bottlenecks = bottlenecks.sort_values(['Priority', 'days_since_updated'], ascending=[False, False])

    return bottlenecks[['Issue key', 'Summary', 'Priority', 'Assignee', 'days_since_updated', 'resolution_time']]


def generate_priority_pie_chart():
    fig = px.pie(df, names='Priority', title='Priority Distribution')
    return json.loads(fig.to_json())

def generate_assignee_bar_chart():
    fig = px.bar(df, x='Assignee', y='Issue key', title='Issues per Assignee')
    return json.loads(fig.to_json())

def generate_scatter_plot():
    fig = px.scatter(df, x='days_since_created', y='resolution_time', title='Resolution Time vs. Days Since Created')
    return json.loads(fig.to_json())

def generate_histogram_plot():
    fig = px.histogram(df, x='resolution_time', title='Histogram of Resolution Time')
    return json.loads(fig.to_json())



def find_stagnant_issues(df):
    
    stagnant_issues = df[(df['Priority_Num'] >= 2) & (df['days_since_updated'] > 5)]
    stagnant_issues = pd.concat([stagnant_issues, df[(df['Priority_Num'] <= 2) & (df['days_since_updated'] > 15)]])
    return stagnant_issues


def find_not_updated_issues(df):
    not_updated_issues = df[(df['days_since_created'] > 10) & (df['Updated'].isna())]
    return not_updated_issues

def detect_outliers(df):
    features = df[['resolution_time', 'days_since_updated']].dropna()
    model = IsolationForest(contamination=0.05)
    model.fit(features)
    df['is_outlier'] = model.predict(features)
    outliers = df[df['is_outlier'] == -1]
    return outliers

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        return redirect(url_for('dashboard', query=query))
    return render_template('index.html')


@app.route('/dashboard/<query>')
def dashboard(query):
    stagnant_issues = find_stagnant_issues(df)
    not_updated_issues = find_not_updated_issues(df)
    outliers = detect_outliers(df)
    
    
    priority_pie = None
    resolution_scatter = None
    assignee_bar = None
    
    if not outliers.empty:
        priority_pie = px.pie(outliers, names='Priority', title='Distribution of Priorities in Outliers')
        resolution_scatter = px.scatter(outliers, x='days_since_updated', y='resolution_time', 
                                        hover_data=['Issue key', 'Summary'],
                                        title='Days Since Updated vs Resolution Time')
        
        assignee_bar = px.bar(df, x='Assignee', y='resolution_time', title='Resolution Time by Assignee')

    
    charts = {
        'priority_pie': json.dumps(priority_pie, cls=plotly.utils.PlotlyJSONEncoder) if priority_pie else None,
        'resolution_scatter': json.dumps(resolution_scatter, cls=plotly.utils.PlotlyJSONEncoder) if resolution_scatter else None,
        'assignee_bar': json.dumps(assignee_bar, cls=plotly.utils.PlotlyJSONEncoder) if assignee_bar else None,
    }

    return render_template('dashboard.html', 
                           query=query, 
                           stagnant_issues=stagnant_issues.to_dict(orient='records'),
                           not_updated_issues=not_updated_issues.to_dict(orient='records'),
                           outliers=outliers.to_dict(orient='records'),
                           charts=charts)

if __name__ == '__main__':
    app.run(debug=True)