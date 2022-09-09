## Módulo 2 Machine Learning
## Linear Regression sin frameworks y librerías
## Eduardo Rodríguez Gil
## A01274913

# Importamos las librerias que vamos a utilizar.
import pandas as pd # La libreria de pandas la utilizamos para leer el archivo csv del dataframe.
import matplotlib.pyplot as plt # La libreria de matplotlib la utilizamos para poder mostrar nuestra gráfica.

df = pd.read_csv('vgsales.csv') # Mandamos a llamar nuestro archivo csv del dataframe.

y = df['Other_Sales'].tolist() # Definimos nuestra y.
x = df[['NA_Sales', 'EU_Sales']].to_numpy().tolist() # Definimos nuestras x.
n = [0,0,0]

__errors__= [] # Variable para almacenar los errores.

## Función hypothesis
# Definimos nuestra hypothesis como h(x) con los parametros actuales.
def hypothesis(n, m):
    acum = 0
    for i in range(len(n)):
        acum = acum + n[i] * m[i] #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
    return acum

## Función de MSE
# Muestra los errores que se generan apartir de los valores de hypothesis,
# y el valor verdadero de y
def erros(n, x, y):
    global __errors__
    error_acum = 0
    
    for i in range(len(x)):
        hyp = (hypothesis(n, x[i]) - y[i])**2 # this error is the original cost function, (the one used to make updates in GD is the derivated verssion of this formula)
        print( "hyp  %f  y %f " % (hyp,  y[i]))
        error_acum = error_acum + hyp
    mean_error_z = error_acum/len(x)
    __errors__.append(mean_error_z)

## Función de Gradient Descent
# Sirve para calcular las variables optimas,
# Apartir del gradient descent algorithm
def gradient_descent(n, x, y, alfa):
    temp = []
    for i in range(len(n)):
        acum = 0
        for j in range(len(x)):
            error = hypothesis(n, x[j]) - y[j]
            acum = acum + error * x[j][i] #Sumatory part of the Gradient Descent formula for linear Regression.
        temp.append(n[i] - acum * (alfa/len(x)))
    return temp

alfa = .01 #  learning rate
for i in range(len(x)):
	if isinstance(x[i], list):
		x[i]=  [1]+x[i]
	else:
		x[i]=  [1,x[i]]

epochs = 0

while True: #  run gradient descent until local minima is reached
    oldz = (n)
    print(n)
    n = gradient_descent(n, x, y, alfa)
    erros(n, x, y) #only used to show errors, it is not used in calculation
    epochs = epochs + 1
    if(oldz == n or epochs == 100):  #  local minima is found when there is no further improvement
        print("sample:")
        print(x)
        print("final params:")
        print(n)
        break

# Imprimimos el error final.
print("Error:")
print(__errors__[-1])

# Gráfica del error.
plt.plot(__errors__)
plt.show()
