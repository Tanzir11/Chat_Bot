import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import random

# Load the dataset
df = pd.read_excel("output.xlsx")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Question'], df['Subtopic'], test_size=0.2, random_state=42)

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Subtopic'].unique()))

# Function to tokenize the question
def tokenize_question(question):
    tokens = tokenizer.tokenize(question)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

# Function to classify the question and provide confidence
def classify_question(question):
    token_ids = tokenize_question(question)
    inputs = tokenizer.encode_plus(question, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    logits = model(input_ids, attention_mask=attention_mask)[0]
    predicted_class = logits.argmax().item()
    confidence = logits.softmax(dim=1)[0, predicted_class].item()
    subtopic = df['Subtopic'].unique()[predicted_class]  # Map index to actual subtopic label
    return subtopic, confidence



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
        if confidence > 0.03:
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
