# Disaster Response Pipeline Project

This is a Udacity Data Scientist Nanodegree project, created for learning and training purposes. The project focuses on building a machine learning pipeline to categorize disaster response messages and deploy a web application for classifying new messages.

## Project Overview

The project involves two main components:
1. **ETL Pipeline**: Extract, transform, and load (ETL) data from messages and categories, clean the data, and store it in an SQLite database.
2. **ML Pipeline**: Train a machine learning model on the data to classify messages into various categories.

## File Structure

- **data/**: Contains the disaster messages, categories datasets, and the process for data cleaning.
  - `disaster_messages.csv`: Raw messages dataset.
  - `disaster_categories.csv`: Categories corresponding to the messages.
  - `process_data.py`: Python script to clean and store data in a database.
- **models/**: Contains the machine learning model and its training pipeline.
  - `train_classifier.py`: Python script to train and save the model.
- **app/**: Contains the web application files, including templates for displaying and classifying new messages.
  - `run.py`: Flask web app to predict categories from new messages.
- **README.md**: This file, which provides instructions and an overview of the project.

## Installation
The project requires Python 3.x and the following libraries:

* pandas
* numpy
* scikit-learn
* SQLAlchemy
* nltk
* Flask
* Plotly
  
## Instructions

### 1. Setting up the database and model
Run the following commands in the project's root directory to set up your database and machine learning model:

'Please provide the filepaths of the messages and categories datasets as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument.
*Example*
  - To run the ETL pipeline that cleans data and stores it in a database:
    ```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
    
  - To run the ML pipeline that trains the classifier and saves it as a pickle file:
    ```
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```
### 2. Running the web application

To launch the web app, follow these steps:

1. Navigate to the `app` directory:
   ```
   cd app
   ```
2. Edit the `run.py` file to update the paths for the database and model file.
3. Run the Flask web app:
  ```
python run.py
  ```
Once the server is running, open your web browser and visit `http://localhost:3001/` to use the web application.

## Screenshots
Here is a screenshot for the website already running :"![image](https://github.com/user-attachments/assets/94a4065c-9e3d-49cc-afb9-c233ce5d4df4)

First the website looks like this: ![image](https://github.com/user-attachments/assets/5a01ecbd-04bc-4b51-b2a7-1ea9d5fa5a6a)

Then after writing your message to start analysing it: Press enter or click on *Classify Message* :![image](https://github.com/user-attachments/assets/004fd2d5-5571-46da-9592-2799a405b875)


## Acknowledgments
This project is part of the Udacity Data Scientist Nanodegree program. As well big thank to the open source libraries and Figure8 for providing the Dataset.
