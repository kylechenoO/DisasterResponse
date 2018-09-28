#   DisasterResponse Project
-   This is a project analyzing message data for disaster response.

###   Project Infomation
-   This project develop under Python 3.6.6. Let's have a look of the directory tree:
<pre>
DisasterResponse/                   --> project directory
├── data                            --> dataset directory(only for notebook)
│   ├── categories.csv
│   └── messages.csv
├── ETL Pipeline Preparation.html   --> html export from notebook
├── ETL Pipeline Preparation.ipynb  --> ETL pipeline notebook
├── LICENSE                         --> license file
├── ML Pipeline Preparation.html    --> html export from notebook
├── ML Pipeline Preparation.ipynb   --> ML pipeline notebook
├── pipeline                        --> pipeline project directory
│   ├── app                         --> the flask app directory
│   │   ├── run.py                  --> flask main run file
│   │   └── templates               --> flask templates directory
│   │       ├── go.html
│   │       └── master.html
│   ├── data                        --> dataset directory for pipline
│   │   ├── disaster_categories.csv
│   │   ├── disaster_messages.csv
│   │   └── process_data.py         --> process_data for ETL pipeline
│   ├── models                      --> models directory
│   │   └── train_classifier.py     --> train_classifier for ML pipeline
│   └── README.md                   --> pipeline README file
└── README.md                       --> main README file

</pre>

-   The requirement pkgs:
nltk                    3.3
pickleshare             0.7.4
pandas                  0.23.2
numpy                   1.14.5
SQLAlchemy              1.2.12
sklearn                 0.0
scipy                   1.1.0

###   Run this Project
1.   cd to the pipeline directory
2.   python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
3.   python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
4.   cd to the pipeline/app directory
5.   python3 run.py
6.   go to http://0.0.0.0:3001/

###   Something Else
-   If you have any trouble when you use the pipeline, at first you need to read the README.md in pipeline/ directory.
-   Maybe you can Email me: kyle@hacking-linux.com
