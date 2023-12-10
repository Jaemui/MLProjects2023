""" Josh Tran 
    ITP-449
    Final Project 
    This ML program wrangles data looking at candidate exoplanets and finds the most optimal ML algorithm using a RSCV pipeline. 
    Using the most optimal ML algorithm, we are able to determine the best attributes for determining exoplanets. 
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import math
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

def main():
    pd.set_option('display.max_columns', None)

    input_file = "cummulative.csv"
    nasa_df = pd.read_csv(input_file, skiprows=41)
    
    #data wrangling 
    nasa_df = nasa_df.drop_duplicates()
    nasa_corr = nasa_df.corr()
    # print(nasa_corr)
    nasa_df = nasa_df.drop([

                            # 'koi_dor', do I drop these for all models or will they be dropped automatically for the pca model?
                            'koi_dor_err1', 
                            'koi_dor_err2',
                            'koi_eccen_err1',
                            'koi_eccen_err2',
                            'koi_sma_err1',
                            'koi_sma_err2',
                            'koi_incl_err1',
                            'koi_incl_err2',
                            'koi_teq_err1',
                            'koi_teq_err2',
                            'koi_period_err1',
                            'koi_period_err2', 
                            'koi_duration_err1',
                            'koi_duration_err2',
                            'koi_prad_err1',
                            'koi_prad_err2',
                            'koi_steff_err1',
                            'koi_steff_err2',
                            'koi_srad_err1', 
                            'koi_srad_err2',
                            'koi_smass_err1',
                            'koi_smass_err2'], axis=1)
    nasa_df = nasa_df.dropna()

    #Converting non-numeric data 
    label_encoder = LabelEncoder()
    nasa_df['koi_disposition_encoded'] = label_encoder.fit_transform(nasa_df['koi_pdisposition'])
    nasa_df = nasa_df.drop(['koi_disposition', 'koi_pdisposition'], axis=1)
    print(nasa_df)

    #decompose the feature vector 
    comp = 3 #based on the # of attributes
    pca = PCA(n_components=comp)


    #Standardizing and Splitting the data set 
    X = nasa_df.drop('koi_disposition_encoded', axis=1).values
    y = nasa_df['koi_disposition_encoded']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_df = pd.DataFrame(X, columns=nasa_df.drop('koi_disposition_encoded', axis=1).columns)
    X_pca = pd.DataFrame(pca.fit_transform(X), index=X_df.index) #pca version 
    
    #replace X with X_pca for the PCA version
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=500, stratify=y)

    #Pipeline object 
    pipe = Pipeline([
    	('estimator', None),  # Placeholder for the estimators
    ])

    #list of hyperparameters and their estimator list
    estimator_list = [
            {
            	'estimator':[KNeighborsClassifier()],
            	'estimator__n_neighbors': range(1, int(1.5*(X_train.shape[0]**0.5)))
            },
	        {	
             	'estimator': [DecisionTreeClassifier()],
             	'estimator__max_depth': range(3, 15),
             	'estimator__criterion': ['entropy', 'gini'],
                'estimator__min_samples_leaf': range(1,10)
            },
            { 
                'estimator': [LogisticRegression(max_iter=1000)]
            },
            {
                'estimator': [SVC()],
                'estimator__kernel': ['rbf'],
                'estimator__C':[0.1, 1, 10, 100],
                'estimator__gamma':[0.1, 1, 10]
            }]
    rscv = RandomizedSearchCV(
        pipe,  # Pipeline object
		param_distributions=estimator_list, # collection of estimators and hyperparams
		scoring='accuracy', # scoring metric
    )
    rscv.fit(X_train, y_train)
    print('best params:\n', rscv.best_params_)
    print('scores:\n', rscv.cv_results_['mean_test_score'])

    #hyperparameters of the optimal non_pca algorithm 
    hyperparams = {
        'max_depth': range(3, 15), #reduced the range due to runtime 
        'criterion': ['entropy', 'gini'],
        'min_samples_leaf': range(1,10) #reduced the range due to runtime
    }
    #GSCV of the optimal non_pca algorithm 
    gscv = GridSearchCV(
		estimator = DecisionTreeClassifier(),  
		param_grid = hyperparams
	)
    gscv.fit(X_train, y_train)
    print('Best hyperparameters:\n', gscv.best_params_)
    
    #hyperparameters of the optimal pca algorithm 
    # hyperparams = {
    #     'n_neighbors': range(1, int(1.5*(X_train.shape[0]**0.5)))
    # }
    # #GSCV of the optimal pca algorithm 
    # gscv = GridSearchCV(
	# 	estimator = KNeighborsClassifier(),  
	# 	param_grid = hyperparams
	# )
    # gscv.fit(X_train, y_train)
    # print('Best hyperparameters PCA:\n', gscv.best_params_)

    # building the non_pca model with best hyperparameters 
    model_dtree = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf= 4, random_state=500)
    model_dtree.fit(X_train, y_train)
    y_pred = model_dtree.predict(X_test)

    #buulding the pca model with the best hyperparameters 
    # model_svc = KNeighborsClassifier(n_neighbors=17)
    # model_svc.fit(X_train, y_train)
    # y_pred_pca = model_svc.predict(X_test)

    # confusion matrix for non pca model 
    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    fig, axes = plt.subplots(1,1)
    cm_disp.plot(ax = axes)
    axes.set_title('NASA Confusion Matrix')
    plt.savefig('NASA CM.png')

    #classfication report for non pca model
    class_rep = classification_report(y_test, y_pred)
    print(class_rep)

    # #confusion matrix for pca model 
    # cm_pca = confusion_matrix(y_test, y_pred_pca)
    # cm_disp_pca = ConfusionMatrixDisplay(confusion_matrix = cm_pca)
    # fig, axes = plt.subplots(1,1)
    # cm_disp_pca.plot(ax = axes)
    # axes.set_title('NASA PCA Confusion Matrix')
    # plt.savefig('NASA PCA CM.png')

    #classfication report pca model
    # class_rep_pca = classification_report(y_test, y_pred_pca)
    # print(class_rep_pca)

    #Permutation importance 
    res = permutation_importance(model_dtree, X, y, n_repeats=10, random_state=0)
    print(res.importances_mean)


if __name__ == '__main__':
    main()



#PCA Version
#SVC Algorithim is best used with C: 100, gamma: 0.1, kernel: rbf 
#               precision    recall  f1-score   support

#            0       0.80      0.85      0.83      1374
#            1       0.83      0.79      0.81      1324

#     accuracy                           0.82      2698
#    macro avg       0.82      0.82      0.82      2698
# weighted avg       0.82      0.82      0.82      2698

#KNeighbors algorithm 
#  {'n_neighbors': 27}
#               precision    recall  f1-score   support

#            0       0.79      0.81      0.80      1374
#            1       0.80      0.77      0.79      1324

#     accuracy                           0.79      2698
#    macro avg       0.79      0.79      0.79      2698
# weighted avg       0.79      0.79      0.79      2698

#Non PCA Version 
#Decision Tree Algorithim 
# {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 9}
#               precision    recall  f1-score   support

#            0       0.81      0.83      0.82      1374
#            1       0.82      0.80      0.81      1324

#     accuracy                           0.82      2698
#    macro avg       0.82      0.81      0.81      2698
# weighted avg       0.82      0.82      0.82      2698
#This is the final model
'''
1. PCA did not improve upon the results significanlty according to the classification report.  
2. The dataset is not very collinear and does not suffer from the curse of dimensionality. 
    Due to the PCA value being so high at 11, it could also be the wrong number of principle components. 
    We would need to loop through a range of PCA values to find the most optimal number of principal components. 
3. The model was able to classify objects equally across labels since the classification reports reveal very similar scores
    between the labels. The Precision, recall, and f1-scores for both labels are nearly identical with up to only a .03 difference. 
4  It seems like koi_prad and koi_dor are the most significant attributes on determining whether an object is an exoplanet according 
    to the permutation importance results. 
5. koi_prad represents the planetary radius of an object. It is the most significant attribute since it represents 
    the size of the object, and earth-sized objects are most likely to be considered exoplanents since they are most likely 
    to contain earth-like attributes. 
'''