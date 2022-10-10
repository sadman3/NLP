import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# nltk.download()
###
## If there is an issue to download the package, use it
## cd /Applications/Python\ 3.9
## ./Install\ Certificates.command
###
# nltk.download('punkt')

paragraph = "hello I am Sadman. Graduate Research Assistant at University of South Florida."

sentences = nltk.sent_tokenize(paragraph)

print(sentences)

words = nltk.word_tokenize(paragraph)

print(words)

stemmer = PorterStemmer()

for word in words:
    if stemmer.stem(word) not in stopwords.words('english'):
        print(stemmer.stem(word))

lemmatizer = WordNetLemmatizer()

for word in words:
    if lemmatizer.lemmatize(word) not in stopwords.words('english'):
        print(lemmatizer.lemmatize(word))

cv = CountVectorizer(max_features=1500)
BagOfWords = cv.fit_transform(sentences).toarray()
print(BagOfWords)

tf_idf = TfidfVectorizer(max_features=1500)
X = tf_idf.fit_transform(sentences).toarray()
print(X)