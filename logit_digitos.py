import numpy as np
import matplotlib.pyplot as plt
#Importamos todos los modulos de scikit learn
import sklearn.datasets  as skdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.metrics import confusion_matrix

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
# Vamos a hacer un split training test

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
train_samples = len(X_train)
clf = LogisticRegression(C=50. / train_samples, penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)


#Coeficientes de regresion
# Este codigo es inspirado y fuertemente influenciado de https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-mnist-py 
coef = clf.coef_.copy()
#Tenemos 10 coeficientes de 8*8
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8, 8), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_title('Beta %i' % i)

plt.savefig('coeficiente.png')

#Matriz de confusion
y_pred = clf.predict(X_test)
conf = confusion_matrix(y_test, y_pred)
print(np.shape(conf))
plt.figure(figsize=(5, 5))
plt.imshow(conf)
for i in range(10):
    for j in range(10):
        plt.text(i,j,conf[i][j])
plt.xlabel('Predict')
plt.ylabel('True')
plt.savefig('confusion.png')