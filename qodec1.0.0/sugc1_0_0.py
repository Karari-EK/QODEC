#Summary Generation Code (sugc1.0.0)
#Generates a summary from flagged text files
#can be modified to have the summary written in a text file
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 29/6/2020
#Last Edited: 29/6/2020
#Editor: Karari
#version: 1.0.0
#Appreciation: All creators of packages used in this code
import os
import re, math
import pandas as pd
import sklearn as sk
import easygui
from collections import Counter




#define relevant variables
ff_path = ''#features file path
uf_path=''#User summary file path
sf_path='' #summary file path
of_path='' #output file path
input_str='' #input string
output_str=''#output string
if_file=''#holds name of original file for use in naming generated file
in_folder=''#holds name of original folder for use in naming generated folder

#intialize variables and instantiate objects
#ff_path=easygui.fileopenbox()#obtain the path to the features file
uf_path=easygui.fileopenbox()#obtain the path to the User summary file

#Generate summary
root_path=uf_path
rootdir = uf_path[0:29]
snt=0
cnt=0
print('Root Folder: '+rootdir)#'C:\pfiles\summary\Text\D0601A'
for subdir, dirs, files in os.walk(rootdir):
     for file in files:
          output_str=''#reset the holder of output to file. Each iteration produces output for it's own file
          root_path=os.path.join(subdir, file)
          in_file=root_path[30:41]
          in_folder=root_path[23:29]
          #print(root_path)
          
          with open(root_path,'r',encoding='UTF8') as fp:
               line=fp.readline()
               
               
               while line:
                    #words=format(query_str.strip())
                    input_str = line.split(' ')
                    
                    if(input_str[0]=='1'):#select summary sentences
                         #build the output string
                         i=input_str.pop(0)
                         j=input_str.pop(0)
                         cnt+=len(input_str)
                         input_str=' '.join(input_str)
                         input_str=input_str.strip()
                         print(input_str)
                         output_str=output_str+input_str[1]
                         snt+=1
                    line=fp.readline()     
          output_str=output_str+'\n'          
          
          fp.close()
     print(output_str)
     print('Number of sentences'+str(snt))
     print('Number of words'+str(cnt))


          
          
          #store the summary in a file (user)in C:\pfiles\summary\user
          #sf_path=easygui.fileopenbox()#obtain the path to the output file
          #sf_path='C:\\pfiles\\summary\\user'+in_folder+'\\'+in_file+'.txt'
          #os.makedirs(os.path.dirname(of_path), exist_ok=True)#create folder that does not exist
          #with open(of_path, 'w',encoding='UTF8') as sf_out:
              #sf_out.write(output_str)
              #sf_out.close()
          
