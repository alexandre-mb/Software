# -*- coding: utf-8 -*-
"""
@author: Eduardo_PC
"""

from matplotlib import pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import pandas as pd
import keras
from shapely.geometry import Polygon
import numpy as np
import time
import math
from keras.applications import ResNet50

#Eu uso dataframe pra importar os dados
#Nele tem o nome da imagem, e os labels
#Foi a única forma que encontrei pra poder fazer o que precisava....existem formas mais fáceis
#de importar os dados
df_treinamento = pd.read_table(r"D:\Pastas_Windows\Desktop\Mestrado\teste\Mapping_treinamento.txt", delim_whitespace=True, names=('id','x_centro','y_centro','largura','altura','theta'))


#Modelo Rede Neural Convolutiva

model = models.Sequential()
model.add(layers.Conv2D(64, (5,5), strides=2, name='1', activation='relu', input_shape=(320,320,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), strides=2, name='2', activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), padding = 'same',  name='3',activation='relu'))
model.add(layers.Conv2D(128, (3,3), padding = 'same',  name='4',activation='relu'))
model.add(layers.Conv2D(128, (3,3), padding = 'same',  name='5',activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.9))
model.add(layers.Dense(256, activation='relu')) #relu mesmo????
model.add(layers.Dropout(0.9))
model.add(layers.Dense(256, activation='relu')) #relu mesmo????
model.add(layers.Dense(5, activation='linear')) 

model.summary()

model.compile(optimizer=optimizers.Nadam(), loss='mean_squared_error')

#Esses ImageDataGenerator deixam a imagem no formato certo. Se você quiser fazer data augmentation
#é por eles que faz. Só que é feito online, por isso eu não usei, já que o meu é regressão.

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
#
test_datagen = ImageDataGenerator(rescale=1./255)
#
batch_size = 4
epochs = 10
target_size = (320,320)

#Aqui são gerados os batches. Ele pega o dataframe que eu defini lá em cima,
#o diretório onde as imagens estão, e divide o que é dado de entrada e o que é label
#
train_generator = train_datagen.flow_from_dataframe(dataframe=df_treinamento, 
                                                    directory=r"D:\Pastas_Windows\Desktop\Mestrado\teste\Imagens_Treinamento", 
                                                    x_col='id',
                                                    y_col=['x_centro','y_centro','largura','altura','theta'], 
                                                    has_ext=True, 
                                                    subset='training',
                                                    class_mode="other", 
                                                    target_size=target_size, 
                                                    batch_size=batch_size)

validation_generator = train_datagen.flow_from_dataframe(dataframe=df_treinamento, 
                                                              directory=r"D:\Pastas_Windows\Desktop\Mestrado\teste\Imagens_Treinamento", 
                                                              x_col='id',
                                                              y_col=['x_centro','y_centro','largura','altura','theta'], 
                                                              has_ext=True, 
                                                              subset='validation',
                                                              class_mode="other", 
                                                              target_size=target_size, 
                                                              batch_size=batch_size)



#
#callbacks = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

t1=time.time()

#Todo o treinamento é feito aqui

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n/batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.n/batch_size)

t2=time.time()
print((t2-t1)/3600)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#Pra salvar o modelo e os pesos
# serialize model to JSON
model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
print("Saved model to disk")

#Essa parte é para o teste

df_teste = pd.read_table(r"D:\Pastas_Windows\Desktop\Mestrado\teste\Mapping_teste.txt", delim_whitespace=True, names=('id','x_centro','y_centro','largura','altura','theta'))

test_generator = test_datagen.flow_from_dataframe(dataframe=df_teste, 
                                                  directory=r"D:\Pastas_Windows\Desktop\Mestrado\teste\Imagens_Teste_IW", 
                                                  x_col='id',
                                                  y_col=None, 
                                                  has_ext=True, 
                                                  class_mode=None, 
                                                  shuffle=False,
                                                  target_size=target_size, 
                                                  batch_size=1)

predict2 = model.predict_generator(test_generator,steps=test_generator.n)


#Parte de avaliação dos resultados


pred=np.zeros((1,len(predict2)))
predang=np.zeros((1,len(predict2)))   
predjac=np.zeros((1,len(predict2))) 

i=0
for i in range(0,len(predict2)):
    edge1 = np.array([predict2[i,0] +predict2[i,2]/2*math.cos(predict2[i,4]*np.pi/180) +predict2[i,3]/2*math.sin(predict2[i,4]*np.pi/180), predict2[i,1] -predict2[i,2]/2*math.sin(predict2[i,4]*np.pi/180) +predict2[i,3]/2*math.cos(predict2[i,4]*np.pi/180)])
    edge2 = np.array([predict2[i,0] -predict2[i,2]/2*math.cos(predict2[i,4]*np.pi/180) +predict2[i,3]/2*math.sin(predict2[i,4]*np.pi/180), predict2[i,1] +predict2[i,2]/2*math.sin(predict2[i,4]*np.pi/180) +predict2[i,3]/2*math.cos(predict2[i,4]*np.pi/180)])
    edge3 = np.array([predict2[i,0] -predict2[i,2]/2*math.cos(predict2[i,4]*np.pi/180) -predict2[i,3]/2*math.sin(predict2[i,4]*np.pi/180), predict2[i,1] +predict2[i,2]/2*math.sin(predict2[i,4]*np.pi/180) -predict2[i,3]/2*math.cos(predict2[i,4]*np.pi/180)])
    edge4 = np.array([predict2[i,0] +predict2[i,2]/2*math.cos(predict2[i,4]*np.pi/180) -predict2[i,3]/2*math.sin(predict2[i,4]*np.pi/180), predict2[i,1] -predict2[i,2]/2*math.sin(predict2[i,4]*np.pi/180) -predict2[i,3]/2*math.cos(predict2[i,4]*np.pi/180)])
    
    #Corretos
    edge5 = np.array([df_teste2.iloc[i,1] +df_teste2.iloc[i,3]/2*math.cos(df_teste2.iloc[i,5]*np.pi/180) +df_teste2.iloc[i,4]/2*math.sin(df_teste2.iloc[i,5]*np.pi/180), df_teste2.iloc[i,2] -df_teste2.iloc[i,3]/2*math.sin(df_teste2.iloc[i,5]*np.pi/180) +df_teste2.iloc[i,4]/2*math.cos(df_teste2.iloc[i,5]*np.pi/180)])
    edge6 = np.array([df_teste2.iloc[i,1] -df_teste2.iloc[i,3]/2*math.cos(df_teste2.iloc[i,5]*np.pi/180) +df_teste2.iloc[i,4]/2*math.sin(df_teste2.iloc[i,5]*np.pi/180), df_teste2.iloc[i,2] +df_teste2.iloc[i,3]/2*math.sin(df_teste2.iloc[i,5]*np.pi/180) +df_teste2.iloc[i,4]/2*math.cos(df_teste2.iloc[i,5]*np.pi/180)])
    edge7 = np.array([df_teste2.iloc[i,1] -df_teste2.iloc[i,3]/2*math.cos(df_teste2.iloc[i,5]*np.pi/180) -df_teste2.iloc[i,4]/2*math.sin(df_teste2.iloc[i,5]*np.pi/180), df_teste2.iloc[i,2] +df_teste2.iloc[i,3]/2*math.sin(df_teste2.iloc[i,5]*np.pi/180) -df_teste2.iloc[i,4]/2*math.cos(df_teste2.iloc[i,5]*np.pi/180)])
    edge8 = np.array([df_teste2.iloc[i,1] +df_teste2.iloc[i,3]/2*math.cos(df_teste2.iloc[i,5]*np.pi/180) -df_teste2.iloc[i,4]/2*math.sin(df_teste2.iloc[i,5]*np.pi/180), df_teste2.iloc[i,2] -df_teste2.iloc[i,3]/2*math.sin(df_teste2.iloc[i,5]*np.pi/180) -df_teste2.iloc[i,4]/2*math.cos(df_teste2.iloc[i,5]*np.pi/180)])
    
    p1 = Polygon([(edge1[0],edge1[1]),(edge2[0],edge2[1]),(edge3[0],edge3[1]),(edge4[0],edge4[1])]) 
    p2 = Polygon([(edge5[0],edge5[1]),(edge6[0],edge6[1]),(edge7[0],edge7[1]),(edge8[0],edge8[1])]) #OS LABELS CORRETOS
    
    jaccard = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)   
    ang=abs(df_teste2.iloc[i,5]-predict2[i,4])
    if ang <= 30:
        predang[0,i]=1
    else:
        predang[0,i]=0
    if jaccard >= 0.25:
        predjac[0,i]=1
    else:
        predjac[0,i]=0
    if ang <= 30 and jaccard >= 0.25:
        pred[0,i]=1
    else:
        pred[0,i]=0
acertos=np.sum(pred)
acertosang=np.sum(predang)
acertosjac=np.sum(predjac)
print(100*acertos/test_generator.n)
print(100*acertosang/test_generator.n)
print(100*acertosjac/test_generator.n)
