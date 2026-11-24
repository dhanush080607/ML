import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart_disease_data.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train,X_test,y_train,y_test = train_test_split(
  X,y,test_size=0.3,random_state=42
)

model = GaussianNB()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test,pred))