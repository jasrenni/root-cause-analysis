# root-cause-analysis
##
JIRA ISSUE ANALYSIS
This repository contains two projects focused on analyzing Jira issue summaries and detecting bottlenecks using various NLP techniques and clustering algorithms and is a web-based application designed to analyze Jira data, identify bottlenecks, and visualize the results through interactive charts and graphs. The application is built using Flask as the web framework, with Neo4j for graph database management, and various data processing libraries to handle and analyze the Jira dataset.
###
PROJECT 1 -Jira Bottleneck Analysis and Visualization Dashboard
####
Features
##
Neo4j Database Integration: Connects to a Neo4j graph database to fetch and manage Jira data.
##
Data Preprocessing: Cleans and preprocesses Jira data, including date conversions, calculation of resolution times, and mapping of priority levels.Bottleneck Identification: Uses machine learning (KMeans and IsolationForest) to identify bottlenecks in Jira issues.
###
Visualizations: Generates interactive visualizations using Plotly, including:
Priority distribution pie chart
Issues per assignee bar chart
Resolution time vs. days since creation scatter plot
Resolution time histogram
###
Stagnant and Not-Updated Issues: Detects issues that have not been updated recently or since creation.
###
Outlier Detection: Identifies outliers in the data using IsolationForest based on resolution time and days since the last update.
Flask-Based Dashboard: Provides an intuitive web-based dashboard to display analysis results.

####
Prerequisites
Python 3.x
Neo4j database (configured and running)
Required Python packages (listed in requirements.txt)

####
Set up Neo4j:
Ensure you have Neo4j installed and running.
Update your Neo4j connection settings in the code (e.g., uri, username, password).

The application will start in debug mode, accessible at http://127.0.0.1:5000/.
####
Usage
#####
Landing Page:
The landing page allows you to input a query to analyze specific Jira issues.
#####
Dashboard:
After submitting a query, you'll be directed to the dashboard, where you can view various charts and tables showing the analysis results, including:
#
Priority Distribution: A pie chart of issue priorities.
#
Issues per Assignee: A bar chart displaying the number of issues assigned to each team member.
#
Resolution Time Analysis: A scatter plot comparing resolution time against days since creation, and a histogram of resolution times.
#
Stagnant and Not-Updated Issues: Tables listing issues that have not been updated recently or since creation.
#
Outliers: Identified outliers in the dataset based on resolution time and last update time.
#####
Libraries and Imports:
#
pandas, numpy: For data manipulation.
#
scikit-learn: For machine learning tasks (KMeans, IsolationForest).
#
py2neo: For Neo4j database connection.
#
plotly: For interactive visualizations.
#
flask: For the web framework.

##
/: Landing page for user query input.
/dashboard/<query>: Dashboard displaying analysis results based on the user's query.

#####
Project 2: Sentence-BERT and Network Graph Analysis
####
Overview
This project uses Sentence-BERT embeddings to analyze issue summaries from a Jira dataset. A similarity matrix is generated, and a network graph is built to detect bottlenecks, where nodes represent issues and edges indicate high similarity.
#####
Key Steps
Text Preprocessing: Cleaning and tokenizing issue summaries.
##
Sentence-BERT Embeddings: Generating vector embeddings using the paraphrase-MiniLM-L6-v2 model from the Sentence-Transformers library.
##
Cosine Similarity Calculation: Measuring the similarity between issues.
##
Graph Construction: Building a network graph using NetworkX to identify bottlenecks.
##
Bottleneck Detection: Identifying high-degree nodes as bottlenecks.
#####
Dependencies
##
Python 3.x
#
Pandas
#
Sentence-Transformers
#
NetworkX
#
Matplotlib
#
NLTK
#
Outputs
A similarity matrix between issue summaries.
A network graph showing bottlenecks (high-degree nodes).



