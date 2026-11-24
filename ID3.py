from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

data = [
['Sunny','Hot','High','False','No'],
['Sunny','Hot','High','True','No'],
['Overcast','Hot','High','False','Yes'],
['Rain','Mild','High','False','Yes'],
['Rain','Cool','Normal','False','Yes'],
['Rain','Cool','Normal','True','No'],
['Overcast','Cool','Normal','True','Yes'],
['Sunny','Mild','High','False','No'],
['Sunny','Cool','Normal','False','Yes'],
['Rain','Mild','Normal','False','Yes'],
['Sunny','Mild','Normal','True','Yes'],
['Overcast','Mild','High','True','Yes'],
['Overcast','Hot','Normal','False','Yes'],
['Rain','Mild','High','True','No']
]

X = [r[:-1] for r in data]
Y = [r[-1] for r in data]

enc = LabelEncoder()
X = [ enc.fit_transform(cols) for cols in zip(*X)]
X = list(zip(*X))

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X,Y)

print(export_text(model, feature_names=['overcast','temp','level','cond']))
print(model.predict([X[0]]))