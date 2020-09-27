#Features Generation Code (fegc1.0.0)
#Generates the features matrix given the preprocessed text (document & topic)
#Where necessary conducts further preprocessing before features generation
#Features generated include :
#Term Frequency(tf), Significant Factor(sf),Temporal Dimansion(td)
#and sentence length penalty (in this case a reward to non verbose sentences)
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

WORD = re.compile(r'\w+')

#returns vectors given strings(sentence & Query)
#designed for cosine similarity/sentence similarity
def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

#returns cosine given string vectors(sentence & Query)
#designed for cosine similarity/sentence similarity
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return round((float(numerator) / denominator),2)

#removes numbers from all strings in a list
#designed for term_frequency
def remove(list): 
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list] 
    return list

#the ratio of number of words in the sentence(input_str) that are also in the query(words) 
#to the total number of words in the sentence
#designed for term_frequency
def term_frequency(input_str, words): 
     cnt=0 #holds number of words in the sentence(input_str) that are also in the query(words)
     i=0 #loop counter
     while i<len(input_str):
          # returns true if a string in index i of the sentence is also in the query(words)
          if input_str[i] in words:
               cnt+=1 
          i+=1
     return round(cnt/float(len(input_str)),2)
"""     
def significant_factor(term_frequency, input_str):
     return round((term_frequency)**2/float(len(input_str)),2)
"""
def temporal_dimension(ny,cy):
     return round(0.05/float(1+ny-cy),2)

def sentence_penalty(input_str):
     x=len(input_str)
     if x>5 and x<20:
          return 0.05
     return 0


#define relevant variables
if_path = ''#input file path
of_path='' #output file path
input_str='' #input string
output_cos=''#holdscosine value
output_str=''#output string
query_path=''#Query String #
query_str=''#Query String #
query_folders=''#number of files in duc_2006 or duc_2007
if_files=''#number of files in cluster folder
in_folder=''#input/sentence file folder

#intialize variables and instantiate objects
if_path=easygui.fileopenbox()#obtain the path to the input file 'C:\pfiles\2007\PPFiles\D0701A'
query_path=easygui.fileopenbox()#obtain the path to the query file
#ny=float(easygui.enterbox('Enter the year of newest document','2000'))#obtain the year of newest document
ny=200
#in_file=if_path[25:36]
in_folder=if_path[23:29]
#prepare the query for genration of features
query_path = format(query_path.strip())
       
#convert query string to one sentence	   
with open(query_path,'r',encoding='UTF8') as fp:  
   line = fp.readline()
   #print(line)
   #cnt = 1
   #snt=1
   sent=''
   #convert the query into one line
   #Query will be compared by each sentence
   while line:
        sent=sent+' '+line
        line=fp.readline()
   sent=sent.replace('\n',' ')
   sent=sent.replace('  ',' ')
   sent=sent.replace('   ',' ')
   query_str=sent.strip()
   #print(query_str)
fp.close()

#generate the cosine similarity
root_path=if_path[0:29]
#root_path=root_dir=if_path[0:29]; it would have been 'C:\pfiles\2007\PPFiles\' if we wanted to iterate over all files at one go
#we need to iterate one folder at a time because each folder has a different query
rootdir = if_path[0:29]
print('Root Folder: '+rootdir)#'C:\pfiles\2007\PPFiles\DO701A'
for subdir, dirs, files in os.walk(rootdir):
     for file in files:
          output_str=''#reset the holder of output to file. Each iteration produces output for it's own file
          root_path=os.path.join(subdir, file)
          in_file=root_path[30:36]
          print(root_path)
          with open(root_path,'r',encoding='UTF8') as fp:
               #words=set(query_str.strip())#create a set of words from dictionary file
               cy=float(root_path[34:36]) #obtain documents cy from file path-used in time dimension
               
               line=fp.readline()
               cnt=0
               snt=1
               while line:
                    words=format(query_str.strip())
                    input_str = line #format(line.strip())
                    line=fp.readline()
                    if(len(input_str)==2):
                         ss=0.0
                         tf=0.0
                         td=temporal_dimension(ny,cy)
                         sp=0.0
                         features_str=str(tf)+' '+str(ss)+' '+str(td)+' '+str(sp)+' '+'\n'
                         output_str=output_str+'Input_file: '+in_file+' :Sentence=: '+ str(snt)+' :Features: '+ features_str
                         snt+=1
                    elif(input_str !=''):
                         #determine the sentence similarity(ss)
                         #use Cosine similarity
                         vector1 = text_to_vector(query_str)
                         vector2 = text_to_vector(input_str)
                         theCosine = get_cosine(vector1, vector2)#cosine similarity
                         ss = str(round(theCosine,2))
                         #print('ss'+str(ss))
                         #determine the Term Frequency(tf)
                         #split the strings into word arrays/word lists
                         input_str=input_str.split(' ')
                         words=words.split(' ')
                         #remove numbers
                         input_str=remove(input_str)
                         words=remove(words)
                         #remove duplicates
                         input_str=sorted(set(input_str), key=lambda x:input_str.index(x))
                         words=sorted(set(words), key=lambda x:words.index(x))           
                         tf=term_frequency(input_str,words)
                         #print('tf'+str(tf))
                         #determine the significant factor (sf)
                         #sf=significant_factor(tf,input_str)
                         #print('sf'+str(sf))
                         #determine the temporal dimension(td)
                         td=temporal_dimension(ny,cy)#ny=year of newest document, cy=current year
                         #print('td'+str(td))
                         #determine sentence length penalty
                         #actually a reward for 'well sized sentences'
                         sp=sentence_penalty(input_str)
                         features_str=str(tf)+' '+str(ss)+' '+str(td)+' '+str(sp)+' '+'\n'
                         output_str=output_str+'Input_file: '+in_file+' :Sentence=: '+ str(snt)+' :Features: '+ features_str
                         snt+=1
                    else:
                         continue

                    line=fp.readline()
               cnt+=1
          #build the output string
          print(output_str) 
          fp.close()

          #store the features matrix in a file (features)in C:\pfiles\features folder
          #of_path=easygui.fileopenbox()#obtain the path to the output file
          of_path='C:\\pfiles\\2007\\features\\'+in_folder+'\\'+in_file+'.txt'
          os.makedirs(os.path.dirname(of_path), exist_ok=True)#create folder that does not exist
          with open(of_path, 'w',encoding='UTF8') as f_out:
              f_out.write(output_str)
              f_out.close()

#EOF
