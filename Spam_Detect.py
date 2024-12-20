import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from sklearn.utils import resample

file_path = './Data/Data8.csv'
data = pd.read_csv(file_path)

data = data[['Body', 'Label']]
data.dropna(inplace=True)

spam = data[data['Label'] == 1]
non_spam = data[data['Label'] == 0]

spam_upsampled = resample(spam, replace=True, n_samples=len(non_spam), random_state=42)
data_balanced = pd.concat([non_spam, spam_upsampled])

data_balanced = data_balanced.sample(frac=2, random_state=42, replace=True ).reset_index(drop=True)

print("Basic Data Exploration:")
print(f"Shape of the provided data: {data.shape}")
print(f"Label Distribution:\n{data['Label'].value_counts()}")

print(f"Shape of the balanced data: {data_balanced.shape}")
print(f"Label Distribution:\n{data_balanced['Label'].value_counts()}")

stop_words = list(stopwords.words('english'))
print("\nTransforming text data with CountVectorizer...")
count_vectorizer = CountVectorizer(stop_words=stop_words)
X_count = count_vectorizer.fit_transform(data_balanced['Body'])
y = data_balanced['Label']

print(f"Shape after CountVectorizer: {X_count.shape}")


print("\nDownscaling data using TF-IDF...")
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_count)

print(f"Shape after TF-IDF: {X_tfidf.shape}")

split_index = int(0.8 * len(data_balanced))
X_train = X_tfidf[:split_index]
X_test = X_tfidf[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print(f"Training Set Size: {X_train.shape[0]}")
print(f"Testing Set Size: {X_test.shape[0]}")

print("\nTraining Naive Bayes Classifier...")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

print("\nPerforming 3-Fold Cross-Validation...")
cv_scores = cross_val_score(nb_classifier, X_train, y_train, cv=3, scoring='accuracy')
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

print("\nEvaluating the model on the test data...")
y_pred = nb_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest Results:")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy on Test Data: {test_accuracy}")

new_comments = [
    "Congratulations! You've won a $1000 gift card! Click here to claim.",
    "Buy one get one free! Limited time offer.",
    "Earn money from home easily. Sign up now!",
    "Click on the link below to claim $1 Million!",
    "Free vacation package! Call now to book.",
    " ",
    "URL: http://www.newsisfree.com/click/-5,8305901,1717/\
    Date: 2002-09-27T09:52:32+01:00(The Japan Times)",
    "Meeting rescheduled to 3 PM. Please update your calendars.",
    "Hope you are doing good.",
    "Hi! This is Rishi"
]

print("\nTesting new comments...")
new_comments_transformed = tfidf_transformer.transform(count_vectorizer.transform(new_comments))
new_predictions = nb_classifier.predict(new_comments_transformed)
for comment, prediction in zip(new_comments, new_predictions):
    print(f"Comment: {comment}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Non-Spam'}\n")
