# NLP Question Classifier

This project demonstrates a simple question classifier using Natural Language Processing (NLP) techniques. The classifier is trained on a dataset of questions and their corresponding subtopics.

## Installation

1. Clone the repository: `git clone https://github.com/Tanzir11/Chat_Bot.git`
2. Navigate to the project directory: `cd Chat_Bot`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Prepare the dataset:
   - Create an Excel file named `output.xlsx` containing the questions and their subtopics.
   - Ensure that the questions are stored in the "Question" column and the corresponding subtopics are in the "Subtopic" column.
2. Run the script: `nltk_word_tokenizer.py`
3. Interact with the bot:
   - Enter your questions when prompted by the "User: " input.
   - The bot will classify the questions into subtopics and provide a confidence score for the prediction.
   - You can end the conversation by entering one of the predefined conversation enders.

## Dependencies

- pandas
- scikit-learn
- nltk
- Bert


## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m "Add your feature"`
4. Push your changes to your branch: `git push origin feature/your-feature`
5. Submit a pull request detailing your changes.

## Authors

- [Tanzir](https://github.com/Tanzir11)

## Acknowledgments

- This project is inspired by the need to classify questions based on subtopics using NLP techniques.
