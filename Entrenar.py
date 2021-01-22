import os
import sys
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K # Para borrar

K.clear_session()
training_data='./data/entrenamiento'
validation_data='./data/validacion'

#Parámetros de la red neuronal

epocas=20
altura, longitud=100,100
batch_size=32
pasos=1000
pasos_validacion=200
filtro_convolucion1=32
filtro_convolucion2=64
size_filtro1=(3,3)
size_filtro2=(2,2)
pool_size1=(2,2)
clases=2
lr=0.0007

# Etapa de preprocesamiento digital

entrenamiento=ImageDataGenerator( # Para el entrenamiento
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
    )
validacion=ImageDataGenerator(rescale=1./255)

entrenamiento_generador = entrenamiento.flow_from_directory(
    training_data,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
    )

validacion_generador = validacion.flow_from_directory(
    validation_data,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
    )

# Creación de la red neuronal

cnn=Sequential()
cnn.add(Convolution2D(filtro_convolucion1,size_filtro1,padding='same',input_shape=(altura,longitud,3),activation='relu'))

cnn.add(MaxPooling2D(pool_size=pool_size1))

cnn.add(Convolution2D(filtro_convolucion2,size_filtro2,padding='same',activation='relu'))

cnn.add(MaxPooling2D(pool_size=pool_size1))

cnn.add(Flatten())

cnn.add(Dense(256,activation='relu'))

cnn.add(Dropout(0.35))
cnn.add(Dense(clases,activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

cnn.fit(entrenamiento_generador,steps_per_epoch=pasos,epochs=epocas, validation_data=validacion_generador,validation_steps=pasos_validacion)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')



