import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

messages = pd.read_csv("smsspamcollection/SMSSpamCollection",
                       sep="\t",
                       names=["label", "message"])

ps = PorterStemmer()

corpus = []
for i in range(len(messages)):
    filtered = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    filtered = filtered.lower()

    filtered = filtered.split()

    filtered = [
        ps.stem(word) for word in filtered
        if word not in stopwords.words('english')
    ]

    filtered = ' '.join(filtered)

    corpus.append(filtered)

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

Y = pd.get_dummies(messages['label'])

Y = Y.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=0)

spam_detection_model = MultinomialNB().fit(X_train, Y_train)
Y_pred = spam_detection_model.predict(X_test)

confusion_m = confusion_matrix(Y_test, Y_pred)
print(confusion_m)

accuracy = accuracy_score(Y_test, Y_pred)
print("accuracy=", accuracy)