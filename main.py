import datetime
import re
import random
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import nltk
import logging
from math import prod
import time
from colorama import Fore,Back,Style
from datetime import datetime

# Record start time
start_time = time.time()
now = datetime.now()
def loggerPrint(text):
    print(f"{str(now)}: [ {Fore.GREEN + "OK" + Style.RESET_ALL} ] {text}...")
    
loggerPrint("Connecting to the Model")
# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log messages to a file named 'app.log'
    ]
)
logger = logging.getLogger(__name__)
logging.info("session started")

# Load data from intents.json
loggerPrint("Load data from intents")
with open('intents.json') as file:
    logging.info("Importing intents")
    data = json.load(file)

# Text Preprocessing
loggerPrint("Text Processing")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(words)

patterns = []
tags = []

for intent in data['intents']:
    loggerPrint("Proccessing patterns")
    for pattern in intent['patterns']:
        patterns.append(preprocess_text(pattern))
        tags.append(intent['tag'])

# Data Augmentation
def augment_data(patterns, tags):
    loggerPrint("Data Augmentation")
    augmented_patterns = []
    augmented_tags = []
    for pattern, tag in zip(patterns, tags):
        augmented_patterns.append(pattern)
        augmented_tags.append(tag)
        words = pattern.split()
        if len(words) > 1:
            random.shuffle(words)
            augmented_patterns.append(' '.join(words))
            augmented_tags.append(tag)
    return augmented_patterns, augmented_tags

patterns, tags = augment_data(patterns, tags)

# Encode the labels
loggerPrint("Data Augmentation")
label_encoder = {label: idx for idx, label in enumerate(set(tags))}
y = [label_encoder[label] for label in tags]

# Split the data
loggerPrint("Spliting the data")
X_train, X_test, y_train, y_test = train_test_split(patterns, y, test_size=0.2, random_state=42, stratify=y)

# Define parameter grids for hyperparameter tuning
loggerPrint("Defining parameter grids for hyperparameter tuning")
nb_param_grid = {
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__min_df': [1, 2, 3],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__alpha': [0.1, 1.0]
}

svm_param_grid = {
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__min_df': [1, 2, 3],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1.0, 10.0],
    'clf__kernel': ['linear', 'rbf']
}

rf_param_grid = {
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__min_df': [1, 2, 3],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20]
}

# Create individual models
loggerPrint("Creating individual model")
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(probability=True))
])

rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# Perform grid search for each model with StratifiedKFold
loggerPrint("Performing grid search")
strat_kfold = StratifiedKFold(n_splits=2)

nb_grid_search = GridSearchCV(nb_pipeline, nb_param_grid, cv=strat_kfold, scoring='accuracy')
nb_grid_search.fit(X_train, y_train)
best_nb_pipeline = nb_grid_search.best_estimator_

svm_grid_search = GridSearchCV(svm_pipeline, svm_param_grid, cv=strat_kfold, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)
best_svm_pipeline = svm_grid_search.best_estimator_

rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=strat_kfold, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
best_rf_pipeline = rf_grid_search.best_estimator_

# Combine models using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[('nb', best_nb_pipeline), ('svm', best_svm_pipeline), ('rf', best_rf_pipeline)],
    voting='soft'
)

# Fit the ensemble model
loggerPrint("Fitting model")
ensemble_model.fit(X_train, y_train)

# Calculate and print model accuracies
def print_model_accuracy(name, model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    # Round to 2 decimal places
    accuracy_percentage = round(accuracy * 100, 2)
    # Convert to string and format
    accuracy_percentage_str = f'{accuracy_percentage:.2f}'
    logging.info(f"{name} Model accuracy: {accuracy_percentage_str}%")
    return accuracy

ensemble_accuracy = print_model_accuracy('Ensemble', ensemble_model, X_test, y_test)

# Response Generation
loggerPrint("Creating Response Generation")
def get_response(tag, data):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def predict_intent(text, pipeline, label_encoder):
    processed_text = preprocess_text(text)
    tag_index = pipeline.predict([processed_text])[0]
    tag = [key for key, value in label_encoder.items() if value == tag_index][0]
    return tag

# Calculate and print the number of parameters
def count_parameters(param_grid):
    return prod(len(v) for v in param_grid.values())

nb_param_count = count_parameters(nb_param_grid)
svm_param_count = count_parameters(svm_param_grid)
rf_param_count = count_parameters(rf_param_grid)

logging.info(f'Number of parameters for Naive Bayes: {nb_param_count}')
logging.info(f'Number of parameters for SVM: {svm_param_count}')
logging.info(f'Number of parameters for Random Forest: {rf_param_count}')

logging.info(f'Total number of parameters: {nb_param_count + svm_param_count + rf_param_count}')

training_end_time = time.time()
logging.info(f"Elapsed training time: {(training_end_time - start_time):.2f} seconds")
# Testing the chatbot

def get_input(user_input):
    if user_input == "":
        pass
    else:
        predicted_tag = predict_intent(user_input, ensemble_model, label_encoder)
        response = get_response(predicted_tag, data)
        return response




