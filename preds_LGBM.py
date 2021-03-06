# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:21:20 2020

@author: Arnaud
"""

###-----------------------------------------------------------------------------------------
### LGBM


i=0
preds_prob=np.zeros((train_rem.shape[0],14))
skf=StratifiedKFold(n_splits=5)
modelazos_lgb=[]
for train_index,val_index in skf.split(train_rem,y_rem):
    K.clear_session()
    i+=1
    print('entrenando modelo número: ',i)
    train=train_rem.loc[train_index,:]
    val=train_rem.loc[val_index,:]
    y_train=y_map_rem[0][train_index]
    y_val=y_map_rem[0][val_index]
    model=lgb.LGBMClassifier(** parametros_rem)
    model.fit(train,y_train,eval_set=[(val,y_val)],early_stopping_rounds=50
                                      eval_metric=lgb_multi_weighted_logloss,verbose=0)
    preds_prob[val_index]=model.predict_proba(val)
    print('mejor model (predict_proba): ',mywloss(to_categorical(y_val),preds_prob[val_index]))
    modelazos_lgb.append(model)    

mywloss(y_cat_rem,preds_prob)  
cm=confusion_matrix(y_map_rem,np.argmax(preds_prob,axis=1))
cm= cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
#

pred_num=np.argmax(preds_prob,axis=1)
sns.countplot(pred_num)


k=0
for modelo in modelazos_lgb:
    k+=1
    print('prediciendo modelo ',k)
    j=0
    for inputeo in inputs_preparados:
        test=pd.read_csv(inputeo)
        j+=1    
        print('prediciendo chunk número ',j)
        object_id=pd.Series(test['object_id'])
        del test['object_id']
        #del test['distmod']
        #Probamos con ss.transform (fiteado en full_train)
#        test[asimetricos]=pt.transform(test[asimetricos])
#        test=ss.transform(test)
        test=pt.transform(test)
        #test=ss.fit_transform(test)
        preds=modelo.predict_proba(test)
        preds_99 = np.ones(preds.shape[0])
        for l in range(preds.shape[1]):
            preds_99 *= (1 - preds[:, l])
        if 'class_99' in class_names:
            class_names.remove('class_99')
        preds_df=pd.DataFrame(preds,columns=class_names)
        preds_df['class_99']=preds_99
#        preds_df['class_99'] = 0.14 * preds_99 / np.mean(preds_99) 
        preds_df['object_id']=object_id
        if j==1:
            preds_df.to_csv(str(k)+'grid_lgb_predicciones.csv', header=True, index=False, float_format='%.6f')
        else:
            preds_df.to_csv(str(k)+'grid_lgb_predicciones.csv', header=False, mode='a', index=False, float_format='%.6f')
        del preds, preds_df, test

    
preds1=pd.read_csv('1grid_lgb_predicciones.csv')    
preds2=pd.read_csv('2grid_lgb_predicciones.csv')    
preds3=pd.read_csv('3grid_lgb_predicciones.csv')
preds4=pd.read_csv('4grid_lgb_predicciones.csv')
preds5=pd.read_csv('5grid_lgb_predicciones.csv')
    

preds_media=pd.DataFrame(columns=preds1.columns)
for c in preds1.columns:
    if c=="object_id":
        preds_media[c]=preds1[c]
    else:
        preds_media[c]=(preds1[c]+preds5[c]+preds2[c]+preds3[c]+preds4[c]) / 5
del  preds2, preds3, preds4, preds5

object_id=preds_media['object_id']
preds_media=normaliza_probs_mal(preds_media)
#preds_media=normaliza_probs(preds_media)

preds_media.to_csv('pred_final_grid_lgb_SMOTE.csv',header=True, index=False)

del preds1,preds_media



pred=pd.read_csv('pred_final_grid_lgb_SMOTE.csv')
del pred['object_id']
pred=np.array(pred)
pred_num=np.argmax(pred,axis=1)
clases=[6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
pred_clase=np.array([clases[i] for i in pred_num])
sns.countplot(pred_clase)
del pred
