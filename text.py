from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

d = fetch_20newsgroups(subset='all', categories=['alt.atheism','sci.space','talk.politics.misc','rec.sport.hockey'])
X_train,X_test,y_train,y_test = train_test_split(
  d.data,d.target,test_size=0.3,random_state=42
)

v = TfidfVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

model = MultinomialNB()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test,pred))