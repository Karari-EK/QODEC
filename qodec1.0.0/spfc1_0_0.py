#Serialization of preprocessed files code (spfc1.0.0)
#Generates serial numbers for sentences per file for all preprocessed files
#Used to compare with test files serialization
#Comparison confirms referential integrity/ no of sentences is maintained after preprocessing
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 29/5/2020
#Last Edited: 29/5/2020
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
if_path = ''#input file path
of_path='' #output file path
input_str='' #input string
output_str=''#output string
if_files=''#number of files in cluster folder
of_filesr=''#input/sentence file folder

#intialize variables and instantiate objects
if_path=easygui.fileopenbox()#obtain the path to the input file: C:\pfiles\PPFiles\D0601A\file.txt
in_folder=if_path[18:24]
print(if_path)
    
#serialize every sentence in a file	   
root_path=if_path
rootdir = if_path[0:18]
print('Root Folder: '+rootdir)#'C:\pfiles\PPFiles\D0601A\'
for subdir, dirs, files in os.walk(rootdir):
     for file in files:
          output_str=''#reset the holder of output to file. Each iteration produces output for it's own file
          root_path=os.path.join(subdir, file)
          in_folder=subdir[18:24]
          in_file=root_path[25:36]
          print(root_path)                   
          with open(root_path,'r',encoding='UTF8') as fp:
               line = fp.readline()
               sent=''
               cnt=1
               #select the line to be serialized
               while line:
                    line=format(line.strip())
                    if(line !=''):
                         sent=str(cnt)+' '+line+'\n\n'
                         output_str=output_str+sent
                         cnt+=1
                    line=fp.readline()
               print(output_str)
          fp.close()
          of_path='C:\\pfiles\\serialized_PPFile\\'+in_folder+'\\'+in_file+'.txt'
          os.makedirs(os.path.dirname(of_path), exist_ok=True)#create folder that does not exist
          with open(of_path, 'w',encoding='UTF8') as f_out:
               f_out.write(output_str)
               f_out.close()



