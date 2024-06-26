import string

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from keras.models import load_model

from sentence_transformers import SentenceTransformer


def clean_text(text: str):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)

    # Tokenize
    text = word_tokenize(text)

    # Stemming
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    # Join tokens back into a single string
    text = " ".join(text)

    return text


def get_embedding(text: str):
    # Load the sentence transformer model
    sentence_transformer_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Preprocess the data
    cleaned_text = clean_text(text)

    # Encode the text data to get embedding
    embedding = sentence_transformer_model.encode(cleaned_text, show_progress_bar=True)

    # Convert embedding to a list of Python floats
    embedding_list = list(map(float, embedding))

    # Convert embedding to a single string representation
    embedding_str = str(embedding_list)

    return embedding_str


def predict_phishing(content: str):
    
    # Load the saved model for phishing email classification
    model = load_model("models/phishing_email_classifier.h5")

    # Encode the preprocessed email text to get its embedding
    email_embedding = get_embedding(content)

    # Process the embedding
    embedding_list = [float(i) for i in email_embedding.strip("[]").split(", ")]

    email_embedding_formatted = np.array([embedding_list])

    # Make a prediction for the embedding
    prediction = model.predict(email_embedding_formatted)

    # Thresholding the prediction
    threshold = 0.5
    return prediction[0][0] >= threshold


# Creating sample mails on my own to test and demonstrate the model
# mails = [
#     (
#         "You won!",
#         "You won the lottery! Please send us your bank account details to claim your prize.",
#     ),
# ]

# for mail in mails:
#     print(mail, predict_phishing(mail[0], mail[1]))
