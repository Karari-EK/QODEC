#Deep Belief Network-Support Vector Machine (DBN-SVM)hybrid Code (svmc1.0.0)
#generates a summary post-clasification ybridization of SVM
#input includes:
#Output from the Deep Belief Network
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 10/7/2020
#Last Edited: 10/7/2020
#Editor: Karari
#version: 1.0.0
#Appreciation: All creators of packages used in this code

import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Define Classification training dataset
df = pd.read_csv('C:\\pfiles\\Matrix\\matrix.csv',header=None, names=['tf','ss','td','sp','us'])

#separate trainig into input and output components
X = np.array(df.drop(['us'], 1))
y = np.array(df['us'])

# fit final model
model = LogisticRegression()
model.fit(X, y)

# Define/Read unclassified data
df = pd.read_csv('C:\\pfiles\\2007\\Matrix\\dbn_svm_matrix.csv',header=None, names=['tf','ss','td','sp'])
#Define the input array
Xnew = np.array(df)
# based on training, make prediction
ynew = model.predict(Xnew)
output_str='' #holds data for output
of_path="C:\\pfiles\\2007\\results\\dbn_svm_output.txt" #the path to the output file
# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%i" % (Xnew[i], ynew[i]))
    output_str=output_str+"X=%s, Predicted=%i \n" % (Xnew[i], ynew[i])
    
ynew=np.reshape(ynew,(3274,1))
Xnew=np.concatenate((Xnew,ynew),axis=1)
print(Xnew)
with open(of_path, 'w',encoding='UTF8') as f_out:
    f_out.write(output_str)
    f_out.close()
#EOF
