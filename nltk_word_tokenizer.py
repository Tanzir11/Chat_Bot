import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import random
nltk.download('punkt')

# Load the dataset
df = pd.read_excel("output.xlsx")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Question'], df['Subtopic'], test_size=0.2, random_state=42)

# Tokenize the text data using word_tokenize from NLTK;
X_train_tokenized = [word_tokenize(question) for question in X_train]
X_test_tokenized = [word_tokenize(question) for question in X_test]

# Convert the tokenized data back to string format
X_train_preprocessed = [' '.join(tokens) for tokens in X_train_tokenized]
X_test_preprocessed = [' '.join(tokens) for tokens in X_test_tokenized]

# Vectorize the preprocessed text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train_preprocessed)
X_test_vectorized = vectorizer.transform(X_test_preprocessed)

# Train the SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_vectorized, y_train)

# Function to classify the question and provide confidence
def classify_question(question):
    question_tokenized = word_tokenize(question)
    question_preprocessed = ' '.join(question_tokenized)
    question_vectorized = vectorizer.transform([question_preprocessed])
    predicted_proba = svm.predict_proba(question_vectorized)
    predicted_subtopic = svm.predict(question_vectorized)
    confidence = predicted_proba.max()
    return predicted_subtopic[0], confidence

# Define a list of default messages
default_messages = [
    "I'm sorry, but I don't have information on that topic.",
    "I'm afraid I can't help you with that question.",
    "That's an interesting question, but it's outside the scope of my knowledge.",
    "I don't have the answer to that. Could you try asking something else?",
]

# Define a list of conversation starters
conversation_starters = [
    "hi",
    "hello",
    "how are you?",
    "how's it going?",
]

# Define a list of conversation enders
conversation_enders = [
    "bye",
    "goodbye",
    "take care",
    "see you later",
]

# Function to generate a response based on user input
def generate_response(user_input):
    if user_input.lower() in conversation_starters:
        return "Bot: Hello! How can I assist you today?"
    elif user_input.lower() == "have a great day":
        return "Bot: Thank you! You have a great day too!"
    elif user_input.lower() in conversation_enders:
        return "Bot: Goodbye! Have a nice day!"
    else:
        subtopic, confidence = classify_question(user_input)
        if confidence > 0.07:
            return f"Bot: Prediction Confidence: {confidence}, Subtopic: {subtopic}"
        else:
            return random.choice(default_messages)

# Start the conversation
print("Bot: Hello! How can I assist you today?")
while True:
    user_input = input("User: ")
    response = generate_response(user_input)
    print(response)
    if user_input.lower() in conversation_enders:
        break
