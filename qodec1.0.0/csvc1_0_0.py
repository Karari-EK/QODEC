#Text File Conversion to CSV code (csvc1.0.0)
#converts .txt files into csv files enmass,
#i.e all files in the 50 folders in one run
#conversion prepares for easier datamanipulation and viewing
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
ff_path=easygui.fileopenbox()#obtain the path to the features file

#Convert text files to CSV files, comma separated
root_path=ff_path
rootdir = ff_path[0:18]

print('Root Folder: '+rootdir)#'C:\pfiles\features\'
for subdir, dirs, files in os.walk(rootdir):
     for file in files:
          output_str=''#reset the holder of output to file. Each iteration produces output for it's own file
          root_path=os.path.join(subdir, file)#C:\pfiles\features\D0649D\filename.txt
          in_file=root_path[26:37]
          in_folder=root_path[19:25]
          print(root_path)
 
          of_path='C:\\pfiles\\features_csv\\'+in_folder+'\\'+in_file+'.csv' #set the output path
          
          df = pd.read_csv(root_path,delimiter=' ')#df=dataframe declared in pandas
          os.makedirs(os.path.dirname(of_path), exist_ok=True)#create folder/directory if its non existent
          
          df.to_csv(of_path, index=False)#write the converted file in a new folder without loss of file name
#End of file
          
    
