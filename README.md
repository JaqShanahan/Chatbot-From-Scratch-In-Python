# Chatbot From Scratch In Python
This repository contains code for a text classification-based chatbot using various machine learning models. The chatbot can predict user intents and generate responses based on the trained model.

## Features

- Text preprocessing including tokenization, lemmatization, and stopwords removal.
- Data augmentation to enhance model performance.
- Model training with Naive Bayes, SVM, and Random Forest classifiers.
- Ensemble model using a Voting Classifier.
- Grid search for hyperparameter tuning.
- Response generation based on user intents.

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/JaqShanahan/Chatbot-From-Scratch-In-Python.git
    cd Chatbot-From-Scratch-In-Python
    ```

2. **Create and activate a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    Ensure you have `pip` installed, and then run:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK resources**:

    The script requires some NLTK resources. Run the following Python commands to download them:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage

1. **Prepare your data**:

    Ensure you have an `intents.json` file in the project directory. This file should follow the structure expected by the code. An example of the JSON structure is:

    ```json
    {
      "intents": [
        {
          "tag": "greeting",
          "patterns": ["Hello", "Hi", "How are you?"],
          "responses": ["Hello!", "Hi there!", "I'm good, thank you!"]
        },
        {
          "tag": "goodbye",
          "patterns": ["Bye", "See you later", "Goodbye"],
          "responses": ["Goodbye!", "See you later!", "Have a nice day!"]
        }
      ]
    }
    ```

2. **Run the script**:

    Execute the script to train the model and perform predictions:

    ```bash
    python Chatbot-From-Scratch-In-Python.py
    ```

3. **Example usage**:

    For an example of how to use the chatbot, refer to the `exampleusage.py` file. This file demonstrates how to interact with the trained model and get responses based on user input.

    ```python
    # exampleusage.py
    import main

    main.get_input("hey")
    ```

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.

## License

This project is licensed under the GNU General Public License (GPL). See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NLTK for text preprocessing and lemmatization.
- Scikit-learn for machine learning models and pipelines.
- Colorama for colored terminal text.

---
