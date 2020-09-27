#RBM Training Data Code (rtdc1.0.0)
#Reads lines from matrices file and converts those whose last
#feature is 1 to an array of arrays for use by RBM as training data 
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 10/7/2020
#Last Edited: 10/7/2020
#Editor: Karari
#version: 1.0.0
#Appreciation: All creators of packages used in this code

#import predefined python packages

import os
import numpy as np
from contextlib import contextmanager
import re, math
import pandas as pd
import sklearn as sk
import easygui
from collections import Counter

@contextmanager
def print_array_on_one_line():
     oldoptions = np.get_printoptions()
     np.set_printoptions(linewidth=np.inf)
     yield
     np.set_printoptions(**oldoptions)


#define relevant variables
mf_path = ''#matrix files path
uf_path=''#User summary file path
sf_path='' #summary file path
of_path='' #output file path
input_str='' #input string
output_str=''#output string
if_file=''#holds name of original file for use in naming generated file
in_folder=''#holds name of original folder for use in naming generated folder


#intialize variables and instantiate objects
mf_path=easygui.fileopenbox()#obtain the path to the matrix file: 'C:\pfiles\Matrix'

#Derive RBM data
df = pd.read_csv(mf_path,delimiter=',', header=None, names=['tf','ss','td','sp','ds'])
#np.set_printoptions(linewidth=np.inf)
x = np.array(df.drop(['ds'], 1))

output_str=x.tolist()
print(output_str)

of_path='C:\\pfiles\\Matrix\\rbm_actual.txt'
os.makedirs(os.path.dirname(of_path), exist_ok=True)#create folder that does not exist
with open(of_path, 'w',encoding='UTF8') as f_out:
     f_out.write(str(output_str))
f_out.close()
