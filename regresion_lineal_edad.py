import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('altura_peso_edad.csv')

print(df.head())

sb.scatterplot(x='Altura', y='Peso', hue='Edad', data=df)

plt.show()

X = df[['Altura','Edad']]
y = df['Peso']


X_procesada = X.values.reshape(-1,2)
y_procesada = y.values.reshape(-1,1)

modelo = LinearRegression()
modelo.fit(X_procesada, y_procesada)

#prediccion con dato conocido
altura = 1.57
edad = 25
prediccion1 = modelo.predict([[altura, edad]])
print(f'El peso estimado para {altura} m de {edad} años es de {prediccion1[0][0]} kg')

print(f'Precision del modelo: {modelo.score(X_procesada, y_procesada)}') 

#print(f"Peso: {modelo.coef_[0][0]}")
#print(f"Sesgo (Bias): {modelo.intercept_[0]}")

#prediccion con dato desconocido
altura_desconocida = 1.62
peso = 47
edad_desconocida = 25
prediccion2 = modelo.predict([[altura_desconocida, edad_desconocida]])
print(f'El peso estimado para {altura_desconocida} m de {edad_desconocida} años es de {prediccion2[0][0]} kg')

# ¿Funciona mejor el modelo? ¿Por qué cree que es así?
# En este caso, dado que la persona con las medidas es mas delgada de lo comun, la precision del modelo no es tan buena, 
# sin embargo, habria que considerar el genero como caracteristica para que el modelo sea mas preciso, 
# debido a que la complexion para mujeres y hombres es distinto, pues aunque esta persona en particular este fuera del
# peso ideal, esta solo un kilo abajo del rango de peso ideal para mujeres.


