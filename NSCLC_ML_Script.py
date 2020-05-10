#%%
#Import all required libraries - 1
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score



#%%
#Analyze content of genomics.csv file - 2
clinical_data = pd.read_csv('clinical.csv')
genomics_data = pd.read_csv('genomics.csv')
unique_genes = genomics_data.Gene.unique()
print(unique_genes)
print("There are {} unique genes among all patient ID's.".format(len(unique_genes)))


# %%
#Combine all genes that are common for each patient identifier on genomics.csv - 3
genomics_data_grouped = genomics_data.groupby('ID')['Gene'].apply(list)
print(genomics_data_grouped)

# %%
#Merging both the clinical and genomic csv files together - 4
merged_dataframes = pd.merge(clinical_data, genomics_data_grouped, on="ID")
print(merged_dataframes)


#%%
#Use the count function in Python to get number of occurrences of particular gene in each row - 5
def count_occurrence(GeneName):
    new_column = []
    for row in merged_dataframes['Gene']:
        if GeneName in row:
            new_column.append(row.count(GeneName))
        else:
            new_column.append(0)
    merged_dataframes[GeneName + '_Counts'] = new_column
    return merged_dataframes

for gene in unique_genes:
    count_occurrence(gene)
    print (merged_dataframes)

#%%
#Determine contents of new merged_dataframes columns - 6
print(list(enumerate(merged_dataframes.columns)))


#%%
#Determine % of NA's missing per column - 7
merged_dataframes_no_genes = merged_dataframes.iloc[:,0:17]
NA_percentage = (len(merged_dataframes_no_genes) - merged_dataframes_no_genes.count()) / len(merged_dataframes_no_genes) * 100.0
print(NA_percentage)


# %%
#Deleting all columns that contain missing values - 8
merged_dataframes = merged_dataframes.drop(['T','N','M','Tumor.Size','Gene'],axis=1)
list(enumerate(merged_dataframes.columns))


#%%
#Drop the gene-related columns that have a sum less than 10
merged_dataframes_only_genes = merged_dataframes.iloc[:,12:62]
gene_sum = merged_dataframes_only_genes.sum(axis=0)
print(gene_sum)
for total_gene_count in gene_sum:
    if total_gene_count >= 10:
        print (total_gene_count)
drop_cols = [12,13,14,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,35,36,37,38,40,41,43,44,45,46,47,48,50,51,52,53,54,55,56,60]
merged_dataframes.drop(merged_dataframes.columns[drop_cols], axis = 1, inplace = True)


#%%
list(enumerate(merged_dataframes.columns))
merged_dataframes.select_dtypes(include=['number'])


# %%
merged_dataframes.Outcome = merged_dataframes.Outcome.map({"Alive": 0, "Dead": 1})
merged_dataframes = pd.get_dummies(merged_dataframes, drop_first=True)
print(list(enumerate(merged_dataframes.columns)))
merged_dataframes['Outcome'].value_counts()


# %%
X = merged_dataframes.drop(['Outcome','ID'], axis=1)
y = merged_dataframes.Outcome
clf = RandomForestClassifier()
clf.fit(X,y)
feature_imp = pd.DataFrame(clf.feature_importances_,index=X.columns)
feature_imp.sort_values(by = 0 , ascending = False)


# %%
features_a = merged_dataframes.iloc[:,[2,3,4,5,7,8,10,19,28,36]]
target_a = merged_dataframes.iloc[:,1]
features_a_train, features_a_test, target_a_train, target_a_test = train_test_split(features_a.values, target_a.values, random_state = 1, stratify=y)
smt = SMOTE()
features_a_train,target_a_train = smt.fit_sample(features_a_train,target_a_train)
np.bincount(target_a_train)

#%%
class Models:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
    def LogisticRegression(self):
        lr =  LogisticRegression()
        lr.fit(self.X_train, self.y_train)
        y_pred1=lr.predict(self.X_test)
        lr_confusion_matrix = confusion_matrix(self.y_test, y_pred1)
        lr_accuracy_score = accuracy_score(self.y_test, y_pred1)
        lr_recall_score = recall_score(self.y_test, y_pred1)
        print(type(y_pred1))
        print("Logistic Regression Predictions: \n{}".format(y_pred1))
        print("Logistic Regression Confusion Matrix: \n{}".format(lr_confusion_matrix))
        print("Logistic Regression Accuracy Score: {}%".format(round(lr_accuracy_score * 100, 0)))
        print("Logistic Regression Recall Score: {}%".format(round(lr_recall_score * 100, 0)))
    def NaiveBayes(self):
        nb =  GaussianNB()
        nb.fit(self.X_train, self.y_train)
        y_pred2=nb.predict(self.X_test)
        nb_confusion_matrix = confusion_matrix(self.y_test, y_pred2)
        nb_accuracy_score = accuracy_score(self.y_test, y_pred2)
        nb_recall_score = recall_score(self.y_test, y_pred2)
        print("\n\nNaive Bayes Predictions: \n{}".format(y_pred2))
        print("Naive Bayes Confusion matrix: \n{}".format(nb_confusion_matrix))
        print("Naive Bayes Accuracy Score: {}%".format(round(nb_accuracy_score * 100, 0)))
        print("Naive Bayes Recall Score: {}%".format(round(nb_recall_score * 100, 0)))
    def DecisionTree(self):
        dt = DecisionTreeClassifier()
        dt.fit(self.X_train, self.y_train)
        y_pred3=dt.predict(self.X_test)
        dt_confusion_matrix = confusion_matrix(self.y_test, y_pred3)
        dt_accuracy_score = accuracy_score(self.y_test, y_pred3)
        dt_recall_score = recall_score(self.y_test, y_pred3)
        print("\n\nDecision Tree Predictions: \n{}".format(y_pred3))
        print("Decision Tree Confusion matrix: \n{}".format(dt_confusion_matrix))
        print("Decision Tree Accuracy Score: {}%".format(round(dt_accuracy_score * 100, 0)))
        print("Decision Tree Recall Score: {}%".format(round(dt_recall_score * 100, 0)))
    def KNN(self):
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(self.X_train, self.y_train)
        y_pred4=knn.predict(self.X_test)
        knn_confusion_matrix = confusion_matrix(self.y_test, y_pred4)
        knn_accuracy_score = accuracy_score(self.y_test, y_pred4)
        knn_recall_score = recall_score(self.y_test, y_pred4)
        print("\n\nKNN Predictions: \n{}".format(y_pred4))
        print("KNN Confusion matrix: \n{}".format(knn_confusion_matrix))
        print("KNN Accuracy Score: {}%".format(round(knn_accuracy_score * 100, 0)))
        print("KNN Recall Score: {}%".format(round(knn_recall_score * 100, 0)))
    def LinearDiscriminantAnalysis(self):
        lda = LinearDiscriminantAnalysis()
        lda.fit(self.X_train, self.y_train)
        y_pred5=lda.predict(self.X_test)
        lda_confusion_matrix = confusion_matrix(self.y_test, y_pred5)
        lda_accuracy_score = accuracy_score(self.y_test, y_pred5)
        lda_recall_score = recall_score(self.y_test, y_pred5)
        print("\n\nLDA Predictions: \n{}".format(y_pred5))
        print("LDA Confusion matrix: \n{}".format(lda_confusion_matrix))
        print("LDA Accuracy Score: {}%".format(round(lda_accuracy_score * 100, 0)))
        print("LDA Recall Score: {}%".format(round(lda_recall_score * 100, 0)))

gc_models = Models(features_a_train, features_a_test, target_a_train, target_a_test)
gc_models_LR = gc_models.LogisticRegression()
gc_models_NB = gc_models.NaiveBayes()
gc_models_DT = gc_models.DecisionTree()
gc_models_KNN = gc_models.KNN()
gc_models_LDA = gc_models.LinearDiscriminantAnalysis()


#%%