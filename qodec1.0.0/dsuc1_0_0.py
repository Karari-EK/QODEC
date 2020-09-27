#Derive Summary Code (dsumc1.0.0)
#determine zero or 1 based on certain sentence features , tf,ss and sp
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 29/6/2020
#Last Edited: 29/6/2020
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

#intialize variables and instantiate objects
#ff_path=easygui.fileopenbox()#obtain the path to the features file
cf_path=easygui.fileopenbox()#obtain the path to the csv file

#Derive summary
root_path=cf_path
rootdir = cf_path[0:22]

print('Root Folder: '+rootdir)#'C:\pfiles\features_csv'
for subdir, dirs, files in os.walk(rootdir):
     for file in files:
          root_path=os.path.join(subdir, file)# 
          print(root_path)
          #convert csv to dataframe and assign headers on the fly
          #in the process adding one more header 'ds' for derived summary
          df = pd.read_csv(root_path,delimiter=',', header=None, names=['1','2','3','4','5','tf','ss','td','sp','ds'])
 
          for i in df.index:
               j=df['tf'][i]
               #code to handle conversion errors where floats are erroneous
               #e.g 0.1.1 or 0.05.1 whic was found to affect ss and sp values in first row only.
               k=str(df['ss'][i])
               l=str(df['sp'][i])
               if(len(l)>4):
                    l=l[0:3]
                    if((l[3:])=='.'):
                         l=l[0:2]
                         
               if(len(k)>4):
                    k=k[0:4]
                    #print(k[3:])
                    if((k[3:])=='.'):
                         k=k[0:2]                    
               l=float(l)
               k=float(k)
               z=(j+k)/2 #compute average of term frequency and sentence similarity
               #deterine whether to accept otr reject the sum (1 or 0) based on this condition
               #the condition is based on various test already caried out yielding 0.086 as the lower average of tf and ss
               #sp is the sentecne penalty and shows that the sentence is fitting "not too long and not too short"
               if(z>0.086 and l>0):
                    x=1
                    df['ds'][i]=x
               else:
                    x=0
                    df['ds'][i]=x
                           
               print(x)
          #write back the data frame to the csv file
          #together with the new  collumn 'ds' for derived summary   
          df.to_csv(root_path, index=False, header=None) 
  #EOF
          
