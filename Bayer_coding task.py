
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
# from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from scipy import interp

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


# reading in both csv files
df_users = pd.read_csv('user.csv')
df_job_desc = pd.read_csv('job_desc.csv')


# In[3]:


# display for quick check
display(df_users.head())
display(df_job_desc.head())


# In[4]:


# extracting basic knowledge about the dataset
if set(df_users['user_id'].unique()) == set(df_job_desc['user_id'].unique()):
    print('Two datasets have same set of unique users')
    if len(df_users) == len(df_job_desc):
        print('Two datasets have same size')
        if len(df_users) == len(df_users['user_id'].unique()):
            print('Both datasets rows are by unique users')
            
print('There are {} unique companies in total'.format(len(df_job_desc['company'].unique())))

# exploring positive label frequency
print('Frequency of Positive Label: {}'.format(df_users['has_applied'].sum()/len(df_users))) 

# exploring feature value range from users.csv
print('Feature range in users.csv: {}-{}'.format(df_users.iloc[:,2:].min().min(), df_users.iloc[:,2:].max().max()))

# exploring 'salary' feature value range from job_desc.csv
print('Salary range in job_desc.csv: {}-{}'.format(df_job_desc['salary'].min(), df_job_desc['salary'].max()))


# In[5]:


# merging the two datasets
df_merged = pd.merge(df_users, df_job_desc.iloc[:,1:], on='user_id')
df_merged = df_merged.set_index('user_id')

# converting categorical variable 'Company' into numerical labels
df_merged['company_encoded'] = LabelEncoder().fit_transform(df_merged['company'])
df_merged = df_merged.drop('company', axis=1)
display(df_merged.head())

# Imputing missing values (NaN)
    # Median Imputation
print('Is there any missing value? {}'.format(df_merged.isnull().any().any()))
print('Imputing missing values...')
for col in df_merged.iloc[:,1:].columns:
    if df_merged[col].isnull().any():
        df_merged.loc[:,col] = df_merged[col].fillna(df_merged[col].median())
print('Is there any missing value now? {}'.format(df_merged.isnull().any().any()))


# In[ ]:


def LR(df):
    # specify target column
    target = 'has_applied'
    
    # define the model
    clf = LogisticRegression(random_state=0, solver='lbfgs')
#     clf = LogisticRegression(solver='liblinear')

    
    # fit the model to explore model coefficients (feature importance)
    df_x = df.drop(target, axis=1)
    df_y = df[target]
    clf.fit(df_x,df_y)

    # examine raw coefficients to obtain feature importance
    feature_importance = pd.Series(clf.coef_[0],index=df_x.columns).sort_values(ascending=False)
#     feature_importance = feature_importance[:10]
    feature_importance = feature_importance.to_frame()
    feature_importance = feature_importance.rename(columns={0: 'raw_coeff'})
    for idx, row in feature_importance.iterrows():
        feature_importance.loc[idx, 'abs_coeff'] = np.abs(row['raw_coeff'])
        feature_importance.loc[idx, 'sign'] = np.sign(row['raw_coeff'])
    feature_importance = feature_importance.sort_values(by='abs_coeff', ascending=False)
    print('Top 10 predictive features: \n',feature_importance[:10],'\n')
    
    # define kfold object before performing k-fold cross validation
    # I'm using 5-fold cross validation in this problem.
    fivefold = KFold(5,True,7) #input params: k, shuffle boolean, random seed
    

    fig = plt.figure()
    colors = ['darkorange', 'cyan', 'magenta', 'yellow', 'green']
    k_counter = 1
    mean_fpr = np.linspace(0,1,100)
    tprs = []
    aurocs = []
    f1s = []
    
    # for each fold, generating/plotting ROC curve and Classification Report
    for train_array, val_array in fivefold.split(df):
        print('---Fold '+str(k_counter)+'\n')
#         display(df.iloc[train])
#         display(df.iloc[test])
        train = df.iloc[train_array]
        val = df.iloc[val_array]
        train_x = train.drop(target,axis=1)
        train_y = train[target]
        val_x  = val.drop(target, axis=1)
        val_y  = val[target]
        
        clf_fit = clf.fit(train_x, train_y)
        
        probs = clf_fit.predict_proba(val_x)
        pos_probs = probs[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(val_y, pos_probs, pos_label=1)
        print('AUROC: {}\n'.format(roc_auc_score(val_y, pos_probs)))
        tprs.append(interp(mean_fpr, fpr, tpr))
        aurocs.append(roc_auc_score(val_y, pos_probs))
        
        preds = clf_fit.predict(val_x)
        f1 = f1_score(val_y, preds, average='micro')
        print('F1 Score: {}\n'.format(f1))
        f1s.append(f1)
        print('Classification Report ::\n'+metrics.classification_report(val_y, preds))
    
        plt.plot(fpr, tpr, color=colors[k_counter-1],
                 lw=1, label='ROC CV'+str(k_counter)+' (AUC = %0.2f)' % roc_auc_score(val_y, pos_probs))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        k_counter += 1
    
    # generating/plotting mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auroc = metrics.auc(mean_fpr, mean_tpr)
    std_auroc = np.std(aurocs)
    plt.plot(mean_fpr, mean_tpr, color='blue',
                 lw=2, label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auroc,std_auroc))
    # plotting the random guess line
    plt.plot([0, 1], [0, 1], color='grey', lw=2, label='Random Guess', linestyle='--')
    plt.legend(loc="lower right", bbox_to_anchor=(1.65, 0.47))
    plt.title('Logistic Regression: ROC for User Application Prediction')
    plt.savefig("AUROC_Logistic Regression_User_Application")
    plt.show()
    feature_importance.to_csv('LR_feature_importance.csv')
    print('Mean F1 score across the folds: {}'.format(np.asarray(f1s).mean()))

start_time = time.time()
LR(df_merged)
print('The model took {} seconds'.format(time.time()-start_time))


# In[9]:


def RF(df):
    
    # specify target column
    target = 'has_applied'
    
    # define the model
    num_trees =200 #70 #90 #30
    clf = RandomForestClassifier(n_estimators=num_trees, min_samples_leaf=10, 
                                     max_features=25, max_depth=50, class_weight='balanced')#max_features="sqrt", min_samples_split=10, max_depth=20, min_samples_leaf=4)#max_depth=30, min_samples_leaf=2) # or just max_depth=20, min_samples_leaf=1
#     if len(df.columns) <10:
#         clf = RandomForestClassifier(n_estimators=num_trees, min_samples_leaf=5, 
#                                      max_features="sqrt", max_depth=20, class_weight='balanced')#max_features="sqrt", min_samples_split=10, max_depth=20, min_samples_leaf=4)#max_depth=30, min_samples_leaf=2) # or just max_depth=20, min_samples_leaf=1
#     else:
#         clf = RandomForestClassifier(n_estimators=num_trees, min_samples_leaf=5, 
#                                      max_features=20, max_depth=20, class_weight='balanced')
    
    # fit the model to explore model coefficients (feature importance)
    df_x = df.drop(target, axis=1)
    df_y = df[target]
    clf.fit(df_x,df_y)

    # examine raw coefficients to obtain feature importance
    feature_importance = pd.Series(clf.feature_importances_[0],index=df_x.columns).sort_values(ascending=False)
#     feature_importance = feature_importance[:10]
    feature_importance = feature_importance.to_frame()
    feature_importance = feature_importance.rename(columns={0: 'raw_coeff'})
    for idx, row in feature_importance.iterrows():
        feature_importance.loc[idx, 'abs_coeff'] = np.abs(row['raw_coeff'])
        feature_importance.loc[idx, 'sign'] = np.sign(row['raw_coeff'])
    feature_importance = feature_importance.sort_values(by='abs_coeff', ascending=False)
    print('Top 10 predictive features: \n',feature_importance[:10],'\n')
    
    # define kfold object before performing k-fold cross validation
    # I'm using 5-fold cross validation in this problem.
    fivefold = KFold(5,True,7) #input params: k, shuffle boolean, random seed
    

    fig = plt.figure()
    colors = ['darkorange', 'cyan', 'magenta', 'yellow', 'green']
    k_counter = 1
    mean_fpr = np.linspace(0,1,100)
    tprs = []
    aurocs = []
    f1s = []
    
    # for each fold, generating/plotting ROC curve and Classification Report
    for train_array, val_array in fivefold.split(df):
        print('---Fold '+str(k_counter)+'\n')
#         display(df.iloc[train])
#         display(df.iloc[test])
        train = df.iloc[train_array]
        val = df.iloc[val_array]
        train_x = train.drop(target,axis=1)
        train_y = train[target]
        val_x  = val.drop(target, axis=1)
        val_y  = val[target]
        
        clf_fit = clf.fit(train_x, train_y)
        
        probs = clf_fit.predict_proba(val_x)
        pos_probs = probs[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(val_y, pos_probs, pos_label=1)
        print('AUROC: {}\n'.format(roc_auc_score(val_y, pos_probs)))
        tprs.append(interp(mean_fpr, fpr, tpr))
        aurocs.append(roc_auc_score(val_y, pos_probs))
        
        preds = clf_fit.predict(val_x)
        f1 = f1_score(val_y, preds, average='micro')
        print('F1 Score: {}\n'.format(f1))
        f1s.append(f1)
        print('Classification Report ::\n'+metrics.classification_report(val_y, preds))
    
        plt.plot(fpr, tpr, color=colors[k_counter-1],
                 lw=1, label='ROC CV'+str(k_counter)+' (AUC = %0.2f)' % roc_auc_score(val_y, pos_probs))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        k_counter += 1
    
    # generating/plotting mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auroc = metrics.auc(mean_fpr, mean_tpr)
    std_auroc = np.std(aurocs)
    plt.plot(mean_fpr, mean_tpr, color='blue',
                 lw=2, label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auroc,std_auroc))
    # plotting the random guess line
    plt.plot([0, 1], [0, 1], color='grey', lw=2, label='Random Guess', linestyle='--')
    plt.legend(loc="lower right", bbox_to_anchor=(1.65, 0.47))
    plt.title('Random Forest: ROC for User Application Prediction')
    plt.savefig("AUROC_Random Forest_User_Application")
    plt.show()
    feature_importance.to_csv('RF_feature_importance.csv')
    print('Mean F1 score across the folds: {}'.format(np.asarray(f1s).mean()))
    
start_time = time.time()
RF(df_merged)
print('The model took {} seconds'.format(time.time()-start_time))


# In[12]:


def GBoost(df):    
    # specify target column
    target = 'has_applied'
    
    # define the model
    clf = GradientBoostingClassifier(n_estimators=200, min_samples_leaf=10, 
                                     max_features=25, max_depth=50)#max_features="sqrt", min_samples_split=10, max_depth=20, min_samples_leaf=4)#max_depth=30, min_samples_leaf=2) # or just max_depth=20, min_sample
#     if len(df.columns) <10:
#         clf = GradientBoostingClassifier(n_estimators=60, min_samples_leaf=5, 
#                                      max_features="sqrt", max_depth=20)#max_features="sqrt", min_samples_split=10, max_depth=20, min_samples_leaf=4)#max_depth=30, min_samples_leaf=2) # or just max_depth=20, min_samples_leaf=1
#     else:
#         clf = GradientBoostingClassifier(n_estimators=60, min_samples_leaf=5, 
#                                      max_features=20, max_depth=20)

    
    # fit the model to explore model coefficients (feature importance)
    df_x = df.drop(target, axis=1)
    df_y = df[target]
    clf.fit(df_x,df_y)

    # examine raw coefficients to obtain feature importance
    feature_importance = pd.Series(clf.feature_importances_[0],index=df_x.columns).sort_values(ascending=False)
#     feature_importance = feature_importance[:10]
    feature_importance = feature_importance.to_frame()
    feature_importance = feature_importance.rename(columns={0: 'raw_coeff'})
    for idx, row in feature_importance.iterrows():
        feature_importance.loc[idx, 'abs_coeff'] = np.abs(row['raw_coeff'])
        feature_importance.loc[idx, 'sign'] = np.sign(row['raw_coeff'])
    feature_importance = feature_importance.sort_values(by='abs_coeff', ascending=False)
    print('Top 10 predictive features: \n',feature_importance[:10],'\n')
    
    # define kfold object before performing k-fold cross validation
    # I'm using 5-fold cross validation in this problem.
    fivefold = KFold(5,True,7) #input params: k, shuffle boolean, random seed
    

    fig = plt.figure()
    colors = ['darkorange', 'cyan', 'magenta', 'yellow', 'green']
    k_counter = 1
    mean_fpr = np.linspace(0,1,100)
    tprs = []
    aurocs = []
    f1s = []
    
    # for each fold, generating/plotting ROC curve and Classification Report
    for train_array, val_array in fivefold.split(df):
        print('---Fold '+str(k_counter)+'\n')
#         display(df.iloc[train])
#         display(df.iloc[test])
        train = df.iloc[train_array]
        val = df.iloc[val_array]
        train_x = train.drop(target,axis=1)
        train_y = train[target]
        val_x  = val.drop(target, axis=1)
        val_y  = val[target]
        
        clf_fit = clf.fit(train_x, train_y)
        
        probs = clf_fit.predict_proba(val_x)
        pos_probs = probs[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(val_y, pos_probs, pos_label=1)
        print('AUROC: {}\n'.format(roc_auc_score(val_y, pos_probs)))
        tprs.append(interp(mean_fpr, fpr, tpr))
        aurocs.append(roc_auc_score(val_y, pos_probs))
        
        preds = clf_fit.predict(val_x)
        f1 = f1_score(val_y, preds, average='micro')
        print('F1 Score: {}\n'.format(f1))
        f1s.append(f1)
        print('Classification Report ::\n'+metrics.classification_report(val_y, preds))
    
        plt.plot(fpr, tpr, color=colors[k_counter-1],
                 lw=1, label='ROC CV'+str(k_counter)+' (AUC = %0.2f)' % roc_auc_score(val_y, pos_probs))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        k_counter += 1
    
    # generating/plotting mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auroc = metrics.auc(mean_fpr, mean_tpr)
    std_auroc = np.std(aurocs)
    plt.plot(mean_fpr, mean_tpr, color='blue',
                 lw=2, label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auroc,std_auroc))
    # plotting the random guess line
    plt.plot([0, 1], [0, 1], color='grey', lw=2, label='Random Guess', linestyle='--')
    plt.legend(loc="lower right", bbox_to_anchor=(1.65, 0.47))
    plt.title('Gradient Boosting: ROC for User Application Prediction')
    plt.savefig("AUROC_Gradient Boosting_User_Application")
    plt.show()
    feature_importance.to_csv('GB_feature_importance.csv')
    print('Mean F1 score across the folds: {}'.format(np.asarray(f1s).mean()))

start_time = time.time()
GBoost(df_merged)
print('The model took {} seconds'.format(time.time()-start_time))


# In[13]:


def SVM(df):
# specify target column
    target = 'has_applied'
    
    # define the model
    clf = SVC(kernel='rbf',probability=True) # if want to measure feature importance
    
    # fit the model to explore model coefficients (feature importance)
    df_x = df.drop(target, axis=1)
    df_y = df[target]
    clf.fit(df_x,df_y)

    # examine raw coefficients to obtain feature importance
#     feature_importance = pd.Series(clf.coef_[0],index=df_x.columns).sort_values(ascending=False)
# #     feature_importance = feature_importance[:10]
#     feature_importance = feature_importance.to_frame()
#     feature_importance = feature_importance.rename(columns={0: 'raw_coeff'})
#     for idx, row in feature_importance.iterrows():
#         feature_importance.loc[idx, 'abs_coeff'] = np.abs(row['raw_coeff'])
#         feature_importance.loc[idx, 'sign'] = np.sign(row['raw_coeff'])
#     feature_importance = feature_importance.sort_values(by='abs_coeff', ascending=False)
#     print('Top 10 predictive features: \n',feature_importance[:10],'\n')
    
    # define kfold object before performing k-fold cross validation
    # I'm using 5-fold cross validation in this problem.
    fivefold = KFold(5,True,7) #input params: k, shuffle boolean, random seed
    

    fig = plt.figure()
    colors = ['darkorange', 'cyan', 'magenta', 'yellow', 'green']
    k_counter = 1
    mean_fpr = np.linspace(0,1,100)
    tprs = []
    aurocs = []
    f1s = []
    
    # for each fold, generating/plotting ROC curve and Classification Report
    for train_array, val_array in fivefold.split(df):
        print('---Fold '+str(k_counter)+'\n')
#         display(df.iloc[train])
#         display(df.iloc[test])
        train = df.iloc[train_array]
        val = df.iloc[val_array]
        train_x = train.drop(target,axis=1)
        train_y = train[target]
        val_x  = val.drop(target, axis=1)
        val_y  = val[target]
        
        clf_fit = clf.fit(train_x, train_y)
        
        probs = clf_fit.predict_proba(val_x)
        pos_probs = probs[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(val_y, pos_probs, pos_label=1)
        print('AUROC: {}\n'.format(roc_auc_score(val_y, pos_probs)))
        tprs.append(interp(mean_fpr, fpr, tpr))
        aurocs.append(roc_auc_score(val_y, pos_probs))
        
        preds = clf_fit.predict(val_x)
        f1 = f1_score(val_y, preds, average='micro')
        print('F1 Score: {}\n'.format(f1))
        f1s.append(f1)
        print('Classification Report ::\n'+metrics.classification_report(val_y, preds))
    
        plt.plot(fpr, tpr, color=colors[k_counter-1],
                 lw=1, label='ROC CV'+str(k_counter)+' (AUC = %0.2f)' % roc_auc_score(val_y, pos_probs))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        k_counter += 1
    
    # generating/plotting mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auroc = metrics.auc(mean_fpr, mean_tpr)
    std_auroc = np.std(aurocs)
    plt.plot(mean_fpr, mean_tpr, color='blue',
                 lw=2, label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auroc,std_auroc))
    # plotting the random guess line
    plt.plot([0, 1], [0, 1], color='grey', lw=2, label='Random Guess', linestyle='--')
    plt.legend(loc="lower right", bbox_to_anchor=(1.65, 0.47))
    plt.title('SVM: ROC for User Application Prediction')
    plt.savefig("AUROC_SVM_User_Application")
    plt.show()
#     feature_importance.to_csv('SVM_feature_importance_v2.csv')
    print('Mean F1 score across the folds: {}'.format(np.asarray(f1s).mean()))

start_time = time.time()
SVM(df_merged)
print('The model took {} seconds'.format(time.time()-start_time))

