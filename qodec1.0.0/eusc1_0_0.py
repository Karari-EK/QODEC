#Extract User Summaries Code(eusc1.0.0)
#Extract seroes and ones from user summary files/focus group files
#Extracted summary per user stored in a text file which is the transfered to an excel file
#The Excel file now becomes the overall user summary file
#The avg of user summary/focus group summaries becomes the offcial user summary values
#Excel fie is genrated manually.
#To genrate the excel file automatically, modify this file.
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
fpf_counter=''#counts the number of files per folder
fpi_counter=''#counts the number of files per individual summarizer
sent=''#count the number of sentences per file
indv=''#name of bacth per currator

#intialize variables and instantiate objects
#ff_path=easygui.fileopenbox()#obtain the path to the features file
sf_path=easygui.fileopenbox()#obtain the path to the csv file

#Generate summary
root_path=sf_path
rootdir = sf_path[0:24]
fpi_counter=0
print('Root Folder: '+rootdir)#'C:\pfiles\summary\user\A'
for subdir, dirs, files in os.walk(rootdir):
     fpf_counter=0
     for file in files:
          #output_str=''#reset the holder of output to file. Each iteration produces output for it's own file
          root_path=os.path.join(subdir, file)
          in_file=root_path[20:31]
          in_folder=root_path[25:31]
          indv=root_path[23:24]
          f_path=root_path[0:31]
          
          #print(root_path)
          with open(root_path,'r',encoding='UTF8') as fp:
               line = fp.readline()
               sent=0
               #select the line to be serialized
               while line:
                    line=format(line.strip())
                    line=line.split(' ')
                    if(line[0]=='1'or line[0]=='0'):
                         output_str+=line[0]+'\n'
                         sent+=1
                         fpf_counter+=1
                         fpi_counter+=1
                    line=fp.readline()
               #print(root_path+' '+'has '+str(sent)+' lines')
          fp.close()
     #print('Folder '+in_folder+' has '+str(fpf_counter)+' lines')

print('Individual '+indv+' has '+str(fpi_counter)+' lines')

#store the summary in a file (user)in C:\pfiles\summary\user\A_summary.txt
#change to B_summary.txt or C or D as appropriate or provide for input of the same
#of_path=easygui.fileopenbox()#obtain the path to the output file

of_path='C:\\pfiles\\summary\\user\\'+"D_summary.txt"
os.makedirs(os.path.dirname(of_path), exist_ok=True)#create folder/file if either doesn't exist
with open(of_path, 'w',encoding='UTF8') as sf_out:
    sf_out.write(output_str)
    sf_out.close()
