# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 17:22:38 2018

@author: zahrakin
"""

"""DATA MODELING"""
import pandas as pd
import numpy as np
import psycopg2

dtrain_norm=pd.read_excel('Training Normalize Data.xlsx')
dtest_norm=pd.read_excel('Testing Normalize Data.xlsx')
target_train=pd.read_excel('Training Target.xlsx')
target_test=pd.read_excel('Testing Target.xlsx')
dpred_norm=pd.read_excel('Prediction Data.xlsx')

"""Mined the Data Model"""
con = psycopg2.connect(dbname= 'biteam', 
                       host= 'prod.cnof6its6vqf.ap-southeast-1.redshift.amazonaws.com', 
                       port = '5439', 
                       user= 'ca_team', 
                       password= 'CA11team11')
cur = con.cursor()
cur.execute("SELECT * FROM ca.zmk_datamart_next_trx;")

#Storing to pandas data frame
data=pd.DataFrame(cur.fetchall())
data.columns=np.transpose([i[0]for i in cur.description])
data.head()
data.shape #Melihat jumlah kolom dan baris

#Closing connection
con.close()
cur.close()

#Checking Duplicate Data
data.duplicated().sum()

"""Preparation"""
mem_no=data["mem_no"] #ambil mem_no as grup 1
#print(mem_no)
main_data=data.drop("mem_no",axis=1).drop("target",axis=1)
#print(main_data)
target=data["target"]
#print(target)

#Split Data
def SplitData(x,y):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)
    return(x_train,x_test,y_train,y_test)
data_train,data_test,target_train,target_test=SplitData(main_data,target)
data_train.shape
data_test.shape
target_train.shape
target_test.shape

"""PreProcessing Modeling"""
"""DATA TRAIN"""
#GET NUMERICAL DATA FROM DATA_TRAIN
def NumericCategoric(filename):
    num=filename._get_numeric_data()
    num=num.fillna(value=0)
    cat=filename.drop(num.columns,axis=1)
    cat=cat.fillna(value='none')
    return(num,cat)
num_train,cat_train=NumericCategoric(data_train)
num_train.shape
cat_train.shape
num_train.isnull().sum()
cat_train.isnull().sum()

####CONVERT CATEGORICAL VARIABLE INTO DUMMY VARIABLE####
def DummyVariable (filename):
    filename=pd.get_dummies(filename)
    return(filename)
cat_train.head()
c_train=DummyVariable(cat_train)    
c_train.shape    
num_train.shape
c_train.isnull().sum()
num_train.isnull().sum()
target_train.isnull().sum()

#USE RESET INDEX FOR PREVENT THEM CONCAT WITH THE DATA
def ConcatData(x,y):
    x.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    xy=pd.concat([x,y],axis=1)
    return(xy)
data_train1=ConcatData(c_train,num_train)
data_train1.shape
data_train1.isnull().sum()

####NORMALISASI DATA TRAIN####
def NormalizeData (filename):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    scaler=scaler.fit_transform(filename)
    return(scaler)
dtrain_norm=NormalizeData(data_train1)
dtrain_norm=pd.DataFrame(dtrain_norm,columns=data_train1.columns)
dtrain_norm.head()
dtrain_norm.isnull().sum()

dtrain_norm.to_excel('Training Normalize Data.xlsx',sheet_name='Sheet1')
target_train.to_excel('Training Target.xlsx',sheet_name='Sheet1')

#data_train2=ConcatData(dtrain_norm,target_train)
#
#####EXPORT DATA TRAIN TO EXCEL###
#data_train2.to_excel('Data Training.xlsx',sheet_name='Sheet1')


"""DATA TEST"""
#GET NUMERICAL DATA FROM DATA_TRAIN
num_test,cat_test=NumericCategoric(data_test)
num_test.shape
cat_test.shape
num_test.isnull().sum()
cat_test.isnull().sum()

####CONVERT CATEGORICAL VARIABLE INTO DUMMY VARIABLE####
c_test=DummyVariable(cat_test)
c_test.shape
c_train.shape

##Menyimpan index training
#from sklearn.externals import joblib
#joblib.dump(c_train.columns,"col.pkl")
#train_column=joblib.load("col.pkl")
#
##Reindex kolom prediksi dengan kolom training
#c_test1=c_test.reindex(columns=train_column,fill_value=0)
#c_test1.shape

#missing_col=set(c_train.columns)-set(c_test.columns)
#missing_col

#USE RESET INDEX FOR PREVENT THEM CONCAT WITH THE DATA
data_test1=ConcatData(c_test,num_test)
data_test1.shape
data_test1.isnull().sum()

####NORMALISASI DATA TRAIN####
dtest_norm=NormalizeData(data_test1)
dtest_norm=pd.DataFrame(dtest_norm,columns=data_test1.columns)
dtest_norm.isnull().sum()

dtest_norm.to_excel('Testing Normalize Data.xlsx',sheet_name='Sheet1')
target_test.to_excel('Testing Target.xlsx',sheet_name='Sheet1')

#data_test2=ConcatData(dtest_norm,target_test)
#data_test2.head()
#data_test2.shape

####EXPORT DATA TRAIN TO EXCEL###
#data_test2.to_excel('Data Testing.xlsx',sheet_name='Sheet1')


"""DATA PREDICTION"""
"""Mined the Data Model"""
con = psycopg2.connect(dbname= 'biteam', 
                       host= 'prod.cnof6its6vqf.ap-southeast-1.redshift.amazonaws.com', 
                       port = '5439', 
                       user= 'ca_team', 
                       password= 'CA11team11')
cur = con.cursor()
cur.execute("SELECT * FROM ca.zmk_datamart_next_trx_test;")

#Storing to panda data frame
data_pred=pd.DataFrame(cur.fetchall())
data_pred.columns=np.transpose([i[0]for i in cur.description])
data_pred.shape

#Closing connection
con.close()
cur.close()

#Checking Duplicate Data
data_pred.duplicated().sum()


"""Preparation"""
mem_no_pred=data_pred["mem_no"] 
mem_no_pred
main_data_pred=data_pred.drop("mem_no",axis=1)
main_data_pred.shape

#GET NUMERICAL DATA FROM MAIN_DATA_PRED
num_pred,cat_pred=NumericCategoric(main_data_pred)
num_pred.shape
cat_pred.shape
num_pred.dtypes
num_pred.isnull().sum()
cat_pred.isnull().sum()

####CONVERT CATEGORICAL VARIABLE INTO DUMMY VARIABLE####
c_pred=DummyVariable(cat_pred)
c_pred.shape

#Menyimpan index training
from sklearn.externals import joblib
joblib.dump(dtrain_norm.columns,"col.pkl")
train_column=joblib.load("col.pkl")

#Reindex kolom prediksi dengan kolom training
c_pred1=c_pred.reindex(columns=train_column,fill_value=0)
c_pred1.shape


#USE RESET INDEX FOR PREVENT THEM CONCAT WITH THE DATA
data_pred1=ConcatData(c_pred1,num_pred)
data_pred1.shape
data_pred1.isnull().sum()

####NORMALISASI DATA TRAIN####
dpred_norm=NormalizeData(data_pred1)
dpred_norm=pd.DataFrame(dpred_norm,columns=data_pred1.columns)
dpred_norm.isnull().sum()

dpred_norm.to_excel('Prediction Data.xlsx', sheet_name='Sheet1')

#data_pred2=ConcatData(mem_no_pred,dpred_norm)
#data_pred2.dtypes
#
#####EXPORT DATA TRAIN TO EXCEL###
#data_pred2.to_excel('Data Real Test.xlsx',sheet_name='Sheet1')


"""Building Modeling Decision Tree"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

def randomized_dtree (filename_x,filename_y):
    model=DecisionTreeClassifier(random_state=123)
    param_grid = {"criterion": ["gini","entropy"],
                  "max_depth": [2, 3, 4, 5, 6, 7, 9, 10],
                  "min_samples_split": [2, 3, 4, 5, 6, 7, 9, 10],
                  "min_samples_leaf": [2, 3, 5, 7, 9, 10, 11, 12, 13],
                  "max_features": [2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17]
                }
    random_dtree=RandomizedSearchCV(estimator=model,
                                  param_distributions=param_grid,
                                  cv = 5,
                                  n_iter = 10,
                                  n_jobs = 1
                                  )
    random_dtree.fit(filename_x,filename_y)
    print(random_dtree.best_score_)
    print(random_dtree.best_params_)
    return(random_dtree)

dtree=randomized_dtree(dtrain_norm,target_train)

def decision_tree (filename_x, filename_y):
    best_dtree=DecisionTreeClassifier(criterion=dtree.best_params_["criterion"], 
                                  max_depth=dtree.best_params_["max_depth"],
                                  min_samples_split=dtree.best_params_["min_samples_split"],
                                  min_samples_leaf=dtree.best_params_["min_samples_leaf"],
                                  max_features=dtree.best_params_["max_features"],
                                  random_state=123
                                    )
    dtree_result=best_dtree.fit(filename_x, filename_y)
    return(dtree_result)

model_dtree=decision_tree(dtrain_norm, target_train)

#Menyimpan Model
from sklearn.externals import joblib
joblib.dump(model_dtree,"decision_tree.pkl")
model_decision_tree=joblib.load("decision_tree.pkl")


###Training Data predicted and Accuracy
def pred_and_acc (filename_x, filename_y):
    predicted=model_dtree.predict(filename_x)
    accuracy=model_dtree.score(filename_x, filename_y)
    print("predicted", predicted)
    print("accuracy score", accuracy)
    return(predicted, accuracy)   
    
dtree_predict_tr, dtree_acc_tr = pred_and_acc(dtrain_norm, target_train)

#Confusion Matrix
def cfmatrix (filename_x, filename_y, filename_z):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import metrics
    cfmatrix_model=metrics.confusion_matrix(filename_y, filename_z)
    accuracy=model_dtree.score(filename_x, filename_y)
    plt.figure(figsize=(9,9))
    sns.heatmap(cfmatrix_model, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size = 15);
    plt.savefig('toy_Digits_ConfusionSeabornCodementor.png')
    return(plt.savefig('toy_Digits_ConfusionSeabornCodementor.png'))

cfmatrix_dtree_tr=cfmatrix(dtrain_norm, target_train, dtree_predict_tr)  

#Model Performance
def Model_Performance (filename_x,filename_y, filename_z):
    from sklearn.metrics import recall_score #Recall or Sensitivity
    recall=recall_score(filename_x,filename_y,average='binary')
    from sklearn.metrics import precision_score #Precision
    precision=precision_score(filename_x,filename_y,average='binary')
    from sklearn.metrics import f1_score #F1 Score
    f1_score=f1_score(filename_x,filename_y, average='binary')
    from sklearn.metrics import roc_auc_score #AUC Score
    auc=roc_auc_score(filename_x,filename_y)
    from sklearn.metrics import log_loss
    log_loss=log_loss(filename_x,model_dtree.predict_proba(filename_z), eps=1e-15, normalize=True,sample_weight=None)
    print("recall", recall)
    print("precision", precision)
    print("f1_score", f1_score)
    print("auc score", auc)
    print("log loss", log_loss)
    return(recall,precision,f1_score,auc,log_loss)

dtree_recall_tr,dtree_precision_tr,dtree_f1_score_tr,dtree_auc_tr,dtree_log_tr=Model_Performance(target_train, dtree_predict_tr,dtrain_norm)


###Testing Data
dtree_predict_ts, dtree_acc_ts = pred_and_acc(dtest_norm, target_test) 

#Confusion Matrix and Model Performance
cfmatrix_dtree_ts=cfmatrix(dtest_norm, target_test, dtree_predict_ts)  

dtree_recall_ts,dtree_precision_ts,dtree_f1_score_ts,dtree_auc_ts,dtree_log_ts=Model_Performance(target_test, dtree_predict_ts,dtest_norm)

###Real Test Data
dtree_predict_rt = model_dtree.predict(dpred_norm)
dtree_pred=pd.DataFrame(dtree_predict_rt)
dtree_pred_new=ConcatData(mem_no_pred,dtree_pred)
dtree_pred_new.shape

#Export to Excel
dtree_pred_new.to_excel('Real Test Decision Tree.xlsx',sheet_name='Sheet1')


"""Building Modeling Random Forest"""
from sklearn.ensemble import RandomForestClassifier

def randomized_ranfor (filename_x,filename_y):
    model=RandomForestClassifier(random_state=123)
    param_grid = {"n_estimators": [3, 5, 7, 9, 10, 11, 12],
                  "criterion": ["gini","entropy"],
                  "min_samples_split": [2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  "min_samples_leaf": [2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15],
                  "max_features": ["sqrt", 2, 3, 5, 6, 7, 8]
                }
    random_ranfor=RandomizedSearchCV(estimator=model,
                                  param_distributions=param_grid,
                                  cv = 5,
                                  n_iter = 10,
                                  n_jobs = 1
                                  )
    random_ranfor.fit(filename_x,filename_y)
    print(random_ranfor.best_score_)
    print(random_ranfor.best_params_)
    return(random_ranfor)

ranforest=randomized_ranfor(dtrain_norm,target_train)

def random_forest (filename_x, filename_y):
    best_ranfor=RandomForestClassifier(n_estimators=ranforest.best_params_["n_estimators"],
                                       criterion=ranforest.best_params_["criterion"],
                                       min_samples_split=ranforest.best_params_["min_samples_split"],
                                       min_samples_leaf=ranforest.best_params_["min_samples_leaf"],
                                       max_features=ranforest.best_params_["max_features"],
                                       random_state=123
                                    )
    ranfor_result=best_ranfor.fit(filename_x, filename_y)
    return(ranfor_result)

model_ranforest=random_forest(dtrain_norm, target_train)

#Menyimpan Model
from sklearn.externals import joblib
joblib.dump(model_ranforest,"random_forest.pkl")
model_ranforest=joblib.load("random_forest.pkl")


###Training Data predicted and Accuracy
def pred_and_acc (filename_x, filename_y):
    predicted=model_ranforest.predict(filename_x)
    accuracy=model_ranforest.score(filename_x, filename_y)
    print("predicted", predicted)
    print("accuracy score", accuracy)
    return(predicted, accuracy)   
    
ranfor_predict_tr, ranfor_acc_tr = pred_and_acc(dtrain_norm, target_train)

#Confusion Matrix
def cfmatrix (filename_x, filename_y, filename_z):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import metrics
    cfmatrix_model=metrics.confusion_matrix(filename_y, filename_z)
    accuracy=model_ranforest.score(filename_x, filename_y)
    plt.figure(figsize=(9,9))
    sns.heatmap(cfmatrix_model, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size = 15);
    plt.savefig('toy_Digits_ConfusionSeabornCodementor.png')
    return(plt.savefig('toy_Digits_ConfusionSeabornCodementor.png'))

cfmatrix_ranfor_tr=cfmatrix(dtrain_norm, target_train, ranfor_predict_tr)  

#Model Performance
def Model_Performance (filename_x,filename_y, filename_z):
    from sklearn.metrics import recall_score #Recall or Sensitivity
    recall=recall_score(filename_x,filename_y,average='binary')
    from sklearn.metrics import precision_score #Precision
    precision=precision_score(filename_x,filename_y,average='binary')
    from sklearn.metrics import f1_score #F1 Score
    f1_score=f1_score(filename_x,filename_y, average='binary')
    from sklearn.metrics import roc_auc_score #AUC Score
    auc=roc_auc_score(filename_x,filename_y)
    from sklearn.metrics import log_loss
    log_loss=log_loss(filename_x,model_ranforest.predict_proba(filename_z), eps=1e-15, normalize=True,sample_weight=None)
    print("recall", recall)
    print("precision", precision)
    print("f1_score", f1_score)
    print("auc score", auc)
    print("log loss", log_loss)
    return(recall,precision,f1_score,auc,log_loss)

ranfor_recall_tr,ranfor_precision_tr,ranfor_f1_score_tr,ranfor_auc_tr,ranfor_log_tr=Model_Performance(target_train, ranfor_predict_tr,dtrain_norm)


###Testing Data
ranfor_predict_ts, ranfor_acc_ts = pred_and_acc(dtest_norm, target_test) 

#Confusion Matrix and Model Performance
cfmatrix_ranfor_ts=cfmatrix(dtest_norm, target_test, ranfor_predict_ts)  

ranfor_recall_ts,ranfor_precision_ts,ranfor_f1_score_ts,ranfor_auc_ts,ranfor_log_ts=Model_Performance(target_test, ranfor_predict_ts,dtest_norm)

###Real Test Data
ranfor_predict_rt = ranforest.predict(dpred_norm)
ranfor_pred=pd.DataFrame(ranfor_predict_rt)
ranfor_pred_new=ConcatData(mem_no_pred,ranfor_pred)
ranfor_pred_new.shape

#Export to Excel
ranfor_pred_new.to_excel('Real Test Random Forest.xlsx',sheet_name='Sheet1')


"""Building Model Gradient Boosting"""
from sklearn.ensemble import GradientBoostingClassifier

def randomized_gradboos (filename_x,filename_y):
    model=GradientBoostingClassifier(random_state=123)
    param_grid = {"n_estimators": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                  "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "max_features": ["sqrt", 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  "learning_rate": [0.05, 0.02, 0.01, 0.5, 0.4, 0.3, 0.2, 0.1]
                }
    random_gradboos=RandomizedSearchCV(estimator=model,
                                  param_distributions=param_grid,
                                  cv = 5,
                                  n_iter = 10,
                                  n_jobs = 1
                                  )
    random_gradboos.fit(filename_x,filename_y)
    print(random_gradboos.best_score_)
    print(random_gradboos.best_params_)
    return(random_gradboos)

gradboos=randomized_gradboos(dtrain_norm,target_train)

def gradient_boosting (filename_x, filename_y):
    best_gradboos=GradientBoostingClassifier(n_estimators=gradboos.best_params_["n_estimators"],
                                       min_samples_split=gradboos.best_params_["min_samples_split"],
                                       max_features=gradboos.best_params_["max_features"],
                                       learning_rate=gradboos.best_params_["learning_rate"],
                                       random_state=123
                                    )
    gradboos_result=best_gradboos.fit(filename_x, filename_y)
    return(gradboos_result)

model_gradboos=gradient_boosting(dtrain_norm, target_train)


###Training Data predicted and Accuracy
def pred_and_acc (filename_x, filename_y):
    predicted=model_gradboos.predict(filename_x)
    accuracy=model_gradboos.score(filename_x, filename_y)
    print("predicted", predicted)
    print("accuracy score", accuracy)
    return(predicted, accuracy)   
    
gradboos_predict_tr, gradboos_acc_tr = pred_and_acc(dtrain_norm, target_train)

#Confusion Matrix
def cfmatrix (filename_x, filename_y, filename_z):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import metrics
    cfmatrix_model=metrics.confusion_matrix(filename_y, filename_z)
    accuracy=model_gradboos.score(filename_x, filename_y)
    plt.figure(figsize=(9,9))
    sns.heatmap(cfmatrix_model, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size = 15);
    plt.savefig('toy_Digits_ConfusionSeabornCodementor.png')
    return(plt.savefig('toy_Digits_ConfusionSeabornCodementor.png'))

cfmatrix_gradboos_tr=cfmatrix(dtrain_norm, target_train, gradboos_predict_tr)  

#Model Performance
def Model_Performance (filename_x,filename_y, filename_z):
    from sklearn.metrics import recall_score #Recall or Sensitivity
    recall=recall_score(filename_x,filename_y,average='binary')
    from sklearn.metrics import precision_score #Precision
    precision=precision_score(filename_x,filename_y,average='binary')
    from sklearn.metrics import f1_score #F1 Score
    f1_score=f1_score(filename_x,filename_y, average='binary')
    from sklearn.metrics import roc_auc_score #AUC Score
    auc=roc_auc_score(filename_x,filename_y)
    from sklearn.metrics import log_loss
    log_loss=log_loss(filename_x,model_gradboos.predict_proba(filename_z), eps=1e-15, normalize=True,sample_weight=None)
    print("recall", recall)
    print("precision", precision)
    print("f1_score", f1_score)
    print("auc score", auc)
    print("log loss", log_loss)
    return(recall,precision,f1_score,auc,log_loss)

gradboos_recall_tr,gradboos_precision_tr,gradboos_f1_score_tr,gradboos_auc_tr,gradboos_log_tr=Model_Performance(target_train, gradboos_predict_tr,dtrain_norm)


###Testing Data
gradboos_predict_ts, gradboos_acc_ts = pred_and_acc(dtest_norm, target_test) 

#Confusion Matrix and Model Performance
cfmatrix_gradboos_ts=cfmatrix(dtest_norm, target_test, gradboos_predict_ts)  

gradboos_recall_ts,gradboos_precision_ts,gradboos_f1_score_ts,gradboos_auc_ts,gradboos_log_ts=Model_Performance(target_test, gradboos_predict_ts,dtest_norm)

###Real Test Data
gradboos_predict_rt = gradboos.predict(dpred_norm)
gradboos_pred=pd.DataFrame(gradboos_predict_rt)
gradboos_pred_new=ConcatData(mem_no_pred,gradboos_pred)
gradboos_pred_new.shape

#Export to Excel
gradboos_pred_new.to_excel('Real Test Gradient Boosting.xlsx',sheet_name='Sheet1')


xgboost
learning rate, max depth, subsample, colsample bytree, min child weight, n estimator

cv5 n iter=10 scoring=scoring njobs1 n estimator 123