import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('altura_peso.csv')

print(df.head())

sb.scatterplot(x='Altura', y='Peso', data=df)

#plt.show()

X = df['Altura']
y = df['Peso']

X_procesada = X.values.reshape(-1,1)
y_procesada = y.values.reshape(-1,1)

modelo = LinearRegression()
modelo.fit(X_procesada, y_procesada)

#prediccion con dato conocido
altura = 1.57
prediccion1 = modelo.predict([[altura]])
print(f'El peso estimado para {altura} m es de {prediccion1[0][0]} kg')

print(f'Precision del modelo: {modelo.score(X_procesada, y_procesada)}') 

#print(f"Peso: {modelo.coef_[0][0]}")
#print(f"Sesgo (Bias): {modelo.intercept_[0]}")

#prediccion con dato desconocido
altura_desconocida = 1.62
peso = 47
prediccion2 = modelo.predict([[altura_desconocida]])
print(f'El peso estimado para {altura_desconocida} m es de {prediccion2[0][0]} kg')

#¿Funciona bien o no? ¿Por qué cree que es así?
# Funciona bien pero no es posible que sea exacto pues esta guiandose de datos estandares 
# y no va a aplicar para todos los casos, especialmente aquellos fuera del rango del peso 
# ideal como ser casos de obesidad o desnutricion.


