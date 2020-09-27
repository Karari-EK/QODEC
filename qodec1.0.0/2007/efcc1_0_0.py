#Extract features from features.csv code (efcc1.0.0)
#Extract all fetures and the summary collun as well from featurs.csv files
#Aggreagates all features into one big matrix in readiness for SVM input
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 9/7/2020
#Last Edited: 9/7/2020
#Editor: Karari
#version: 1.0.0
#Appreciation: All creators of packages used in this code

#import predefined python packages

import os
import re, math
import pandas as pd
import sklearn as sk
import easygui
from collections import Counter




#define relevant variables
cf_path = ''#csv files path
uf_path=''#User summary file path
sf_path='' #summary file path
of_path='' #output file path
input_str='' #input string
output_str=''#output string
if_file=''#holds name of original file for use in naming generated file
in_folder=''#holds name of original folder for use in naming generated folder
fpf_counter=''#counts the number of files per folder
fpi_counter=''#counts the number of files per individual summarizer
sent=''#count the number of sentences per file
indv=''#name of bacth per currator

#intialize variables and instantiate objects
#ff_path=easygui.fileopenbox()#obtain the path to the features file
cf_path=easygui.fileopenbox()#obtain the path to the csv file

#Derive summary
root_path=cf_path
rootdir = cf_path[0:27]
k=0
j=0
print('Root Folder: '+rootdir)#'C:\pfiles\2007\features_csv'
for subdir, dirs, files in os.walk(rootdir):
     l=0
     for file in files:
          root_path=os.path.join(subdir, file)#
          in_folder=root_path[23:29]
          #print(root_path)
          #convert csv to dataframe and assign headers on the fly
          #access the collumn called ds for its values
          df = pd.read_csv(root_path,delimiter=',', header=None, names=['1','2','3','4','5','tf','ss','td','sp','ds'])
          features=''
          #convert the overall sumarry into dataframe
          #acess the last collumn now labelled dt(derived total) and write the figures to it.
          #df1= pd.read_csv('C:\\pfiles\\summary\\user\\summary.csv',delimiter=',', header=None, names=['A','B','C','D','T','DT'])
          for i in df.index:
               a=str(df['tf'][i])#obtain the 0 or 1 in derived summary (ds) collumn
               b=str(df['ss'][i])#obtain the 0 or 1 in derived summary (ds) collumn
               c=str(df['td'][i])#obtain the 0 or 1 in derived summary (ds) collumn
               d=str(df['sp'][i])#obtain the 0 or 1 in derived summary (ds) collumn
               s=str(df['ds'][i])#obtain the 0 or 1 in derived summary (ds) collumn

               if(len(a)>4):
                    a=a[0:4]
                    #print(k[3:])
                    if((a[3:])=='.'):
                         a=a[0:2]
                 

               if(len(b)>4):
                    b=b[0:3]
                    if((b[3:])=='.'):
                         b=b[0:2]
                         
               if(len(c)>4):
                    c=c[0:4]
                    #print(k[3:])
                    if((c[3:])=='.'):
                         c=c[0:2]

               if(len(d)>4):
                    d=d[0:4]
                    #print(k[3:])
                    if((d[3:])=='.'):
                         d=d[0:2]
                 

               features=str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(s)
               output_str+=features+'\n'
               k+=1
               l+=1
               #print(features)
          #print("File "+root_path+" has "+str(i+1)+" lines")
          
     #print("Folder "+in_folder+" has "+str(l+1)+" lines")
#print("Total Lines "+str(k)+" of 33525 written")
          
print("Done-Check file in path:C:\\pfiles\\Matrix\\matrix.txt")
#write back the data frame df1 to the csv file
#together with the new  collumn 'DT' for derived total
#store the resolved feature matrix in a file (matrix)in C:\pfiles\Matrix folder
#of_path=easygui.fileopenbox()#obtain the path to the output file
of_path='C:\\pfiles\\2007\\Matrix\\matrix.txt'
#os.makedirs(os.path.dirname(of_path), exist_ok=True)#create folder that does not exist
with open(of_path, 'w',encoding='UTF8') as f_out:
              f_out.write(output_str)
              f_out.close()
#pd.to_csv('C:\\pfiles\\summary\\user\\s.csv', index=False, header=None) 
  #EOF
