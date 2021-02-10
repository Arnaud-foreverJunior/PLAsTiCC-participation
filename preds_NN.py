# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:01:43 2020

@author: Arnaud
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:47:19 2020

@author: Arnaud
"""

checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=0)
skf=StratifiedKFold(n_splits=5,shuffle=True)
modelos_NN=[]




preds_prob=np.zeros((train_rem.shape[0],14))
i=0
for train_index,val_index in skf.split(train_rem,y_rem):
    K.clear_session()
    i+=1
    print('entrenando modelo número: ',i)    
    train=train_rem.loc[train_index,:]
    val=train_rem.loc[val_index,:]
    y_train=y_cat_rem[train_index]
    y_val=y_cat_rem[val_index]
    model=build_model(activation='tanh',size=train_rem.shape[1])
    checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=0)
    model.compile(loss=mywlossNN, optimizer="sgd", metrics=['accuracy'])
    history=model.fit(train,y_train,validation_data=[val,y_val],batch_size=100,epochs=600,verbose=0,callbacks=[checkPoint])
    model.load_weights("./keras.model")
    history_df=pd.DataFrame(history.history)
    history_df.loc[5:,['loss','val_loss']].plot()
    history_df.loc[5:,['accuracy','val_accuracy']].plot()
    preds_prob[val_index]=model.predict_proba(val)
    print('mejor model (predict_proba): ',mywlossNN(y_val,preds_prob[val_index,:]))
    modelos_NN.append(model    )
    model.save('NeuralNetwork/modelo' + str(i) +'.h5')
    
cm=confusion_matrix(y_map_rem,np.argmax(preds_prob,axis=1))
cm= cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]


k=0
for modelo in modelos_NN:
    k+=1
    print('prediciendo modelo ',k)
    j=0
    for inputeo in inputs_preparados:
        test=pd.read_csv(inputeo)
        j+=1    
        print('prediciendo chunk número ',j)
        object_id=pd.Series(test['object_id'])
        del test['object_id']
#       test[asimetricos]=pt.transform(test[asimetricos])
#       test=ss.transform(test)
        test=pt.transform(test)
        preds=modelo.predict_proba(test)
        preds_99 = np.ones(preds.shape[0])
        for l in range(preds.shape[1]):
            preds_99 *= (1 - preds[:, l])
        if 'class_99' in class_names:
            class_names.pop(-1)
        preds_df=pd.DataFrame(preds,columns=class_names)
        preds_df['class_99'] =preds_99
        preds_df['object_id']=object_id
        if j==1:
            preds_df.to_csv(str(k)+'predicciones.csv', header=True, index=False, float_format='%.6f')
        else:
            preds_df.to_csv(str(k)+'predicciones.csv', header=False, mode='a', index=False, float_format='%.6f')
        del preds, preds_df, test

preds1=pd.read_csv('1predicciones.csv')    
preds2=pd.read_csv('2predicciones.csv')    
preds3=pd.read_csv('3predicciones.csv')
preds4=pd.read_csv('4predicciones.csv')
preds5=pd.read_csv('5predicciones.csv')
    

preds_media=pd.DataFrame(columns=preds1.columns)
for c in preds1.columns:
    if c=="object_id":
        preds_media[c]=preds1[c]
    else:
        preds_media[c]=(preds1[c]+preds5[c]+preds2[c]+preds3[c]+preds4[c]) / 5
del  preds2, preds3, preds4, preds5




preds_media=normaliza_probs_mal(preds_media)

preds_media.to_csv('pred_final_NN_SMOTE.csv',header=True, index=False)

del preds1,preds_media
        
#ver como van esas predicciones
pred=pd.read_csv('pred_final_NN_SMOTE.csv')
del pred['object_id']
pred=np.array(pred)
pred_num=np.argmax(pred,axis=1)
clases=[6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
pred_clase=np.array([clases[i] for i in pred_num])
sns.countplot(pred_clase)
del pred



