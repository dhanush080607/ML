from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(
    iris.data, iris.target, test_size=0.3,
    random_state=42, stratify=iris.target)

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(Xtr, ytr)

pred = model.predict(Xte)

print("Accuracy:", accuracy_score(yte, pred) * 100)