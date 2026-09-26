📧 Spam Email Classifier

A Machine Learning-based Spam Email/SMS Classification system built using Python, Scikit-learn, TF-IDF, and Multinomial Naive Bayes. The project classifies messages as either Spam or Not Spam (Ham) based on their text content.

📌 Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, or suspicious links.

This project uses Natural Language Processing (NLP) and Machine Learning to automatically identify spam messages.

Workflow
Message
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Multinomial Naive Bayes
   ↓
Spam / Not Spam
🚀 Features
✅ SMS/Email text classification
✅ NLP-based text processing
✅ TF-IDF feature extraction
✅ Multinomial Naive Bayes classification
✅ Train/test dataset splitting
✅ Accuracy evaluation
✅ Classification report
✅ Confusion matrix
✅ Custom message prediction
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Multinomial Naive Bayes
NLP
📊 Dataset

The project uses the SMS Spam Collection Dataset, containing approximately 5,574 labelled messages classified as:

ham → Not Spam
spam → Spam

Dataset source: Kaggle – SMS Spam Collection Dataset

📂 Project Structure
Spam_Email_Classifier/
│
├── spam_classifier.py
├── requirements.txt
├── README.md
│
└── dataset/
    └── spam.csv
⚙️ Installation
1. Clone the Repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd Spam_Email_Classifier
2. Create Virtual Environment
python -m venv venv
3. Activate Virtual Environment

Windows:

venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
▶️ Run the Project

Make sure spam.csv is inside the dataset folder.

Then run:

python spam_classifier.py
📈 Expected Output

The program displays:

Dataset loaded successfully!
Total messages: 5572

==================================================
SPAM EMAIL CLASSIFIER
==================================================

Accuracy: 96.xx%

It also displays:

Classification Report
Precision
Recall
F1-score
Confusion Matrix
🧪 Example Predictions
Example 1
Message:
Congratulations! You have won a free iPhone. Click now!

Prediction:
SPAM
Example 2
Message:
Hi, please send me the project report by evening.

Prediction:
NOT SPAM
Example 3
Message:
URGENT! You have won a cash prize. Claim now!

Prediction:
SPAM
🧠 Machine Learning Model
TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) converts text messages into numerical features that can be processed by a machine learning model.

Multinomial Naive Bayes

The Multinomial Naive Bayes algorithm is used to classify the TF-IDF features into two categories:

0 → NOT SPAM
1 → SPAM
📊 Model Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

These metrics help measure how accurately the classifier identifies spam and legitimate messages.

🔮 Future Improvements
Create a graphical user interface.
Build a Flask web application.
Add real-time email classification.
Improve text preprocessing.
Compare multiple ML algorithms.
Deploy the model as a web service.
🎯 Learning Outcomes

Through this project, I gained practical experience in:

Natural Language Processing
Text classification
TF-IDF vectorization
Machine Learning
Naive Bayes
Model evaluation
Data preprocessing
Spam detection
👨‍💻 Author

Harshal Karad

B.E. – Artificial Intelligence & Data Science

⭐ Project

Spam Email Classifier using Machine Learning

An NLP-based machine learning project that automatically classifies messages as Spam or Not Spam using TF-IDF and Multinomial Naive Bayes.