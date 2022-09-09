## Módulo 2 Machine Learning
## Linear Regression con frameworks
## Eduardo Rodríguez Gil
## A01274913

# Importamos las librerias que vamos a utilizar.
import pandas as pd # La libreria de pandas la utilizamos para leer el archivo csv del dataframe.
import matplotlib.pyplot as plt # La libreria de matplotlib la utilizamos para poder mostrar nuestra gráfica.
import numpy as np # La libreria de numpy la utilizamos para nuestros array.
from sklearn.linear_model import LinearRegression #La libreria de sklearn.linear_model la utilizamos para hacer nuestra Linear Regression.
from sklearn.model_selection import train_test_split # La libreria de sklearn.model_selection la utilizamos para hacer el train y el test.

df = pd.read_csv('vgsales.csv') # Mandamos a llamar nuestro archivo csv del dataframe.

x = df[['Global_Sales', 'NA_Sales']] # Definimos nuestras variables
x.columns = ['Global_Sales', 'NA_Sales'] # Utilizamos el columns para mostrar solo las columnas que definimos

X = np.array(x['Global_Sales']).reshape(-1, 1) # Definimos nuestra variable X en un array
Y = np.array(x['NA_Sales']).reshape(-1, 1) # Definimos nuestra variable Y en un array

# Descartamos cualquier fila sin valor y definimos nuestro test_size
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.50)

# Utilizamos sklearn.linear_model para realizar la Linear Regression
linear_reg = LinearRegression()

# Dividimos los datos en datos en train y test y imprimimos nuestro score
linear_reg.fit(X_train, Y_train)
print(linear_reg.score(X_test, Y_test))

# Utilizamos predict para X_test y luego imprimimos nuestras predicciones
predict = linear_reg.predict(X_test)
print(predict) # Imprimimos predict

# Graficamos la Linear Regression
plt.scatter(X_test, Y_test, color = 'r')
plt.plot(X_test, predict, color = 'k')
plt.show() # Mostramos la gráfica