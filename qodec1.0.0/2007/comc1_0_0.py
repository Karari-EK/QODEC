#Confusion Matrix Code (comc1.0.0)
#Generates the confusion matrix (cm)
#cm genaration based on Actual (user summary) and Predicted (system summary)
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 10/7/2020
#Last Edited: 10/7/2020
#Editor: Karari
#version: 1.0.0
#Appreciation: All creators of packages used in this code

#add predefined code to module code
import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sn

#define dataframes corresponding to each summary
df1 = pd.read_csv('C:\\pfiles\\2007\\confusion matrix\\svm_actual.csv')
df2 = pd.read_csv('C:\\pfiles\\2007\\confusion matrix\\dbn_svm_predicted.csv')

y_actual= np.array(df1)
y_predicted= np.array(df2)
y_actual= y_actual.flatten()
y_predicted= y_predicted.flatten()
#print(y_predicted)

#data = {'y_predicted': y_actual, 'y_actual': y_predicted}
#print(data)

#df = pd.DataFrame(data, columns=['y_actual','y_predicted'])
#confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)
print("Confusion matrix is WIP!")
confusion_matrix = pd.crosstab(y_actual, y_predicted, rownames=['Actual'], colnames=['Predicted'], margins = True)
print(confusion_matrix)
#sn.heatmap(confusion_matrix, annot=True)
