import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import random
nltk.download('punkt')


df = pd.read_excel("output.xlsx")

X_train, X_test, y_train, y_test = train_test_split(df['Question'], df['Subtopic'], test_size=0.2, random_state=42)

X_train_tokenized = [' '.join(sent_tokenize(question)) for question in X_train]
X_test_tokenized = [' '.join(sent_tokenize(question)) for question in X_test]

X_train_preprocessed = X_train_tokenized
X_test_preprocessed = X_test_tokenized

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train_preprocessed)
X_test_vectorized = vectorizer.transform(X_test_preprocessed)

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_vectorized, y_train)

def classify_question(question):
    question_vectorized = vectorizer.transform([question])
    predicted_proba = svm.predict_proba(question_vectorized)
    predicted_subtopic = svm.predict(question_vectorized)
    confidence = predicted_proba.max()
    return predicted_subtopic[0], confidence

def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summarized_text = summarizer(text, max_length=50, min_length=10, do_sample=False)
    return summarized_text[0]['summary_text']

default_messages = [
    "I'm sorry, but I don't have information on that topic.",
    "I'm afraid I can't help you with that question.",
    "That's an interesting question, but it's outside the scope of my knowledge.",
    "I don't have the answer to that. Could you try asking something else?",
]

conversation_starters = [
    "hi",
    "Hey",
    "hey",
    "Hello",
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
        return "Bot: Hey"
    elif user_input.lower() in ["hey", "hey"]:
        return "Bot: Thank you! You have a great day too!"
    elif user_input.lower() == "have a great day":
        return "Bot: Thank you! You have a great day too!"
    elif user_input.lower() in conversation_enders:
        return "Bot: Goodbye! Have a nice day!"
    else:
        subtopic, confidence = classify_question(user_input)
        # if confidence > 0.04:
        #     return f"Bot: Prediction Confidence: {confidence}, Subtopic: {subtopic}"
        # else:
        #     summarized_input = summarize_text(user_input)
        #     subtopic, confidence = classify_question(summarized_input)
        if confidence > 0.07:
                summarized_input = summarize_text(user_input)
                subtopic, confidence = classify_question(summarized_input)
                return f"Bot: Prediction Confidence: {confidence}, Subtopic: {subtopic}\nSummarized Input: {summarized_input}"
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
