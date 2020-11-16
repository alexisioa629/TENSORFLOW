#Librerias
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

#Funcion para predecir clase
def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  print (result)
#Clases a clasificar, se toman del array ->Dependen del indice
  if answer == 0:
    print("pred: Avión")
  elif answer == 1:
    print("pred: Barco")
  elif answer == 2:
    print("pred: Tren")


  return answer

print("Ejemplo 1 (Avión): ")
predict("avion.jpg")
print("\n")
print("Ejemplo 2 (Barco): ")
predict("barco.jpg")
print("\n")
print("Ejemplo 3 (Tren): ")
predict("tren.jpg")