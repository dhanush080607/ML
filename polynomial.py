from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = load_diabetes()
X = data.data[:,2:3]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size = 0.2, random_state = 42
)

poly = PolynomialFeatures(degree=4)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)

print(r2_score(y_test, pred),mean_squared_error(y_test,pred))