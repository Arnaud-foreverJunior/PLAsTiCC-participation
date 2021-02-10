# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:06:36 2020

@author: Arnaud
"""
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import math
import copy
import random

#import pyplot as plt
#from keras.layers import *
#from keras.models import Model, load_model
#from keras.optimizers import Adam, Nadam, SGD
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout, Input
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from keras import regularizers

#from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV


import lightgbm as lgb

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids
#import tensorflow as tf

#parametros
chunks = 5000000
class_weight= {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1}







#Carga de datos
datos=pd.read_csv('input/training_set.csv')
metadatos_train=pd.read_csv('input/training_set_metadata.csv')
##datos_test=pd.read_csv('input/test_set.csv', chunk_size=chunks)
metadatos_test=pd.read_csv('input/test_set_metadata.csv')





#Preprocesado de los datos, a estudiar
def prepara_datos(datos,metadatos):
    datos['flux_ratio_sq'] = np.power(datos['flux'] / datos['flux_err'], 2.0)
    datos['flux_by_flux_ratio_sq'] = datos['flux'] * datos['flux_ratio_sq']
    aggs = {
            'mjd': ['min', 'max', 'size'],
            'passband': ['mean', 'std'],
            'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
            'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
            'detected': ['mean'],
            'flux_ratio_sq':['sum','skew'],
            'flux_by_flux_ratio_sq':['sum','skew'],
            }
    
    agg_train = datos.groupby('object_id').agg(aggs)
    new_columns = [
            k + '_' + agg for k in aggs.keys() for agg in aggs[k]
            ]
    agg_train.columns = new_columns
    agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']
    agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
    agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
    agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
    agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']
    del agg_train['mjd_min'], agg_train['mjd_max'], 
    merged_train = agg_train.reset_index().merge(
    right=metadatos,
    how='left',
    on='object_id'
    )
    #quitamos porque alta correlacion con otras variables, flur_err_min se podria salvar si eso
    del merged_train['flux_err_min'],merged_train['flux_err_max'],merged_train['flux_max'],merged_train['flux_std']
    #quitamos hostgal_specz porque falta muy a menudo en el test, además, no parece aportar mucha información más de la que
    #aporta hostgal_photoz 
    del  merged_train['hostgal_specz']#, merged_train['hostgal_photoz'],merged_train['hostgal_photoz_err'], merged_train['distmod']
    del merged_train['ra'], merged_train['decl'], merged_train['gal_l'],merged_train['gal_b']
    rellena_datos_faltantes(merged_train)

    if 'target' in merged_train:
        del merged_train['object_id']
        y = merged_train['target']
        del merged_train['target']
        return merged_train,y
    if not('target' in merged_train):
        return merged_train
    


class_names = ['class_6',
 'class_15',
 'class_16',
 'class_42',
 'class_52',
 'class_53',
 'class_62',
 'class_64',
 'class_65',
 'class_67',
 'class_88',
 'class_90',
 'class_92',
 'class_95']


    
#devuelve las columnas a las que les faltan datos
def que_datos_faltan(full_train):
    faltan={}
    for c in full_train.columns:
        faltantes=sum(pd.isnull(full_train[c]))
        if faltantes > 0:
            faltan[c]=faltantes
    print('faltan las siguientes clases: ',faltan)
    return faltan

#la distmod es el modulo de la distancia, creo que de momento lo vamos a rellenar con la media y ya
def rellena_datos_faltantes(datos):
    train_mean=41.4321200
    for c in que_datos_faltan(datos).keys():
        datos[c]=datos[c].fillna(train_mean)


def datos_asimetricos(full_train):
    asimetricos=[]
    for c in full_train.columns:
        if (full_train[c].skew() > (max(full_train[c]) - min(full_train[c]))/5):
            asimetricos.append(c)
    print('Las siguientes features son bastante asimétricas', asimetricos)
    return asimetricos
#En el conjunto de entrenamiento los datos asimetricos son:
#['passband_std', 'flux_err_skew', 'detected_mean', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'mwebv', 'galactic']



#Remuestreo, esta vez OVERSAMPLING a todo, UNDERSAMPLING a 90

#primero data set con solo las clases a remuestrear
clases_rem=[6,15,16,42,52,53,62,65,64,67,88,92,95]
def remuestrea(datos,target,clases_over=clases_rem):

    y=np.array(target)
    indices=[y[i] in clases_rem for i in range(y.shape[0])]
    y=pd.DataFrame(y)
    y_a_remuestrear=y.loc[indices]
    a_remuestrear=datos.loc[indices]


    #segundo, aquellos datos que no vamos a remuestrear (d momento)
    y=np.array(y)
    indices_no_rem=[bool(1-(y[i] in clases_rem)) for i in range(y.shape[0])]
    y=pd.DataFrame(y)
    y_no_remuestrear=y.loc[indices_no_rem]
    no_remuestrear=datos.loc[indices_no_rem]

    #tercer, remuestrear aquellos que se deben remuestrear
    SMOTE=SMOTETomek()
    oversampleado,y_oversampleado=SMOTE.fit_resample(a_remuestrear,y_a_remuestrear)

    #Cuarto, unir el remuestreado con lo que no se tenia que remuestrear
    new_train_rem=pd.concat([no_remuestrear,oversampleado],axis=0)
    new_y_rem=pd.concat([y_no_remuestrear,y_oversampleado],axis=0)
    del no_remuestrear, oversampleado, y_no_remuestrear, y_oversampleado
    #Quinto, Undersamplear el grande
    CC=ClusterCentroids()
    train_rem,y_rem=CC.fit_resample(new_train_rem,new_y_rem)
    del new_train_rem
#    #Seis, comprueba que está todo bien:
#    train_rem['target']=y_rem
#    train_rem['galactic']= train_rem['hostgal_photoz']< -0.656046
#    ax = sns.countplot("target", hue="galactic", data=train_rem)
#    del train_rem['target'],train_rem['galactic']
    return train_rem,y_rem
    
  
#datos y procesado que se utilizan
    
#Uncomment and run if full_train.csv is not available
#full_train, y= prepara_datos(datos,metadatos_train)
#full_train.to_csv('full_train.csv',header=True,index=False)
    

full_train=pd.read_csv('full_train.csv')
y=metadatos_train['target']
features=full_train.columns
pt=PowerTransformer(method='yeo-johnson')
full_train_ss=pd.DataFrame(pt.fit_transform(full_train),columns=features)
train_rem,y_rem=remuestrea(full_train_ss,y)


classes=y.unique()
class_map={sorted(classes)[i]:i for i in range(len(classes))}
y_map_rem=pd.DataFrame([class_map[i] for i in y_rem[0]])
y_cat_rem=to_categorical(y_map_rem[0])
y_map=pd.DataFrame([class_map[i] for i in y])
y_cat=to_categorical(y_map)
wtable=np.unique(y_map,return_counts=True)[1] / y_map.shape[0]



# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true,y_pred): 
    yc=y_pred.clip(min=1e-15,max=1-1e-15)
    loss=-(np.mean(np.mean(y_true*np.log(yc),axis=0)/wtable))
    #wtable is a numpy 1d array with (the number of times class y_true occur in the data set)/(size of data set)
    return loss

def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False






#Validacion cruzada con mywloss
def validacion_cruzada(modelo=lgb.LGBMClassifier(), n=5,full_traineo=full_train,y_mapeo=y_map):
    skf=StratifiedKFold(n_splits=n)
    media_perdida=0
    i=0
    y_cat=to_categorical(y_mapeo)
    for train_index,test_index in skf.split(full_traineo,y_mapeo):
        i+=1 
        full_traineo=np.array(full_traineo)
        print('iteración nº ', i)
        train=full_traineo[train_index,:]
        test=full_traineo[test_index,:]
        #Nota: y_train y_test no estan en el mismo formato puesto que cada uno se usa para cada cosa.
        #y_train es una columna con numeros entre el 0 y el 14, uno para cada clase
        #y_test es una codificacion one_hot de y
        y_train=y_mapeo[0][train_index]        
        modelo.fit(train,y_train)                    
        preds=(modelo.predict_proba(test))
        y_test=y_cat[test_index,:]
        perdida=mywloss(y_test,preds)
        media_perdida += perdida
    media_perdida = media_perdida/ n
    return media_perdida




#---------------------------------------------------------------------------
### Prediction Matrix
        
def prediccion_cruzada(modelo, n=8):
    skf=StratifiedKFold(n_splits=n)
    i=0
    preds=np.zeros((full_train.shape[0],14))
    for train_index,test_index in skf.split(full_train,y):
        i+=1 
        print('iteración nº ', i)
        train=full_train.loc[train_index]
        test=full_train.loc[test_index]
        #Nota: y_train y_test no estan en el mismo formato puesto que cada uno se usa para cada cosa.
        #y_train es una columna con numeros entre el 0 y el 14, uno para cada clase
        #y_test es una codificacion one_hot de y
        y_train=y_map.loc[train_index]
        modelo.fit(train,y_train)
        preds[test_index]=(modelo.predict_proba(test))        
    return preds
        
#el plt y el sns no estan importados, la idea de esta funcion es ejecutarla en un notebook.        
def plot_confusion_matrix(preds):
    cm = confusion_matrix(y_map,np.argmax(preds,axis=1))  
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]        
    plt.figure(figsize=(14,14))
    sns.heatmap(cm,annot=True)        
    #añadir que se diga que cosa es cada cosa, leyenda


            

def descategoriza(y_cat):
    y_mapeado=pd.DataFrame(y_cat)
    y_mapeado['map']=sum(y_mapeado[i]*i for i in range(y_mapeado.shape[1]))
    y_mapeado=pd.Series(y_mapeado['map'])
    return y_mapeado
          
del datos, metadatos_train
inputs=['test_set_batch1.csv','test_set_batch2.csv','test_set_batch3.csv','test_set_batch4.csv','test_set_batch5.csv',
       'test_set_batch6.csv','test_set_batch7.csv','test_set_batch8.csv','test_set_batch9.csv','test_set_batch10.csv',
       'test_set_batch11.csv']

inputs_preparados=['preparado_' + a for a in inputs]


parametros_rem={'learning_rate': 0.03, 'min_data_in_leaf': 100, 'num_leaves': 32}
parametros_rem['n_estimators']=1000



#Neural Network
start_neurons=512
activation='tanh'
dropout_rate=0.5

def mywlossNN(y_true,y_pred): 
    #clip_by_value: Recorta los valores del tensor a un mínimo y un máximo especificados.
    #se hace para que no haya ceos creo yo
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.math.log(yc),axis=0)/wtable))
    #wtable is a numpy 1d array with (the number of times class y_true occur in the data set)/(size of data set)
    return loss

K.clear_session()
#Modelo 1: Red Neuronal Simple
def build_model(activation='tanh',dropout_rate=0.5,size=22):
    
    model=Sequential()
    
    model.add(Dense(start_neurons, activation=activation, input_shape=[size]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
        
    model.add(Dense(start_neurons//2,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
        
    model.add(Dense(start_neurons//4,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
        
    model.add(Dense(start_neurons//8,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate/2))
        
    model.add(Dense(14 , activation='softmax'))
    return model


