#Topics Prepartion and Preprocessing Code (tppc1.0.0)
#Topics Preparation converts HTML code to Text
#Preprecessing preprares text for NLP
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 22/5/2020
#Last Edited: 22/5/2020
#Editor: Karari
#version: 1.0.0
#Appreciation: All creators of packages used in this code

#import predefined python packages
import re
import string
import easygui
import html2text
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize,sent_tokenize

#define variables, Objects and Handles
stemmer= PorterStemmer()#stemmer used to invoke methods used in stemming
h = html2text.HTML2Text()#h used to invoke html to text procesing methods
h.ignore_links = True #conversion to text removes all html links <a></a> etc


#define function stemword
def stemword(w):
    #invoke the stemmer.stem and return the stemmed word 
    return (stemmer.stem(w))

#select the input file
in_path=easygui.fileopenbox()#define path of the input file (html or .0123 type)
in_file=in_path[:-5]#extract file name from file path
print('File :'+in_file+' submitted for processing')

#remove HTML code
html_file = open(in_path, 'r')#open html file
source_code = html_file.read()#read html file
text_file=h.handle(source_code)#remove all html code 

#convert text to sentences(this might be unique per file.don't just copy paste)
#print(text_file)
text_file=text_file.replace('  ','##')#mark paragraphs
#separate topic number into individual sentences
text_file=text_file.replace('##D0',' ##D0')#mark paragraphs
text_s=text_file.split()
#print(text_s)
y=''
i=0
u=len(text_s)
while i<u:
    if text_s[i].startswith('D0') or text_s[i].startswith('##D0'):
        x=text_s[i].split('##')
        #print(x)
        j=0
        while(j<len(x)):
            if x[j].startswith('D0'):
                 x[j]='#'+x[j]+'#'
            j+=1
        y=' '.join(x)
        y=y.split('\n')
        #print(y)
        text_s[i]=' '.join(y)
    i+=1
text_file=' '.join(text_s)

text_file=' '.join(text_s)
#print(text_file)

text_file=text_file.replace('\n',' ')#remove unwanted/illegal paragraphs
text_file=text_file.replace('. ','#')#mark sentences that are also end of paragraphs
text_file=text_file.replace('? ','#')#mark sentences that are also end of paragraphs
text_file=text_file.replace(' #','#')#mark sentences that are also end of paragraphs
text_file=text_file.replace('# ','#')#mark sentences that are also end of paragraphs
text_file=text_file.replace('.S#','.S ')#mark sentences that are also end of paragraphs
text_file=text_file.replace('###','#')#remove empty sentences/paragraphs
text_file=text_file.replace('##','#')#""
text_file=text_file.replace('#','.\n\n')#create a paragraph for each sentence
#text_file=text_file.replace('\n ','\n')#remove leading spaces in sentences
#print(text_file)

#progress communication: Content for generating human summary ready
print('file: '+in_file+' Processed for generation of human summary')

#obtains the path where the file will be saved
out_path=easygui.filesavebox(msg='Confirm or change details of your file', title='Save Document', default=in_file+'.txt', filetypes=' \*.txt')
#saves the file in th eprovided path
with open(out_path, 'w',encoding='UTF8') as f:
    f.write(text_file)
print('Document file: '+in_file+' Saved in '+out_path)#dispalys to the user the location of the saved file

#Text Preprocessing
print('File :'+in_file+' submitted for preprocessing')
#change end of sentence character to #
#Enables reconstruction of sentences after stop words removal
text_file=text_file.replace('.\n\n','#')
text_file = re.sub('[^a-zA-Z0-9\n#]', ' ', text_file)#remove special and unwanted characters
text_file=text_file.lower()#convert all text to lower case; facilitates stop word removal

#remove stop words and converts words to their stem/root
stop_words = set(stopwords.words('english'))# define stopwords object 
word_tokens = word_tokenize(text_file)#split text to words, word_tokens is a list
#removes all stopwords from word tokens, filtered_sentence is a list

#Alternative code for removal of stopwords using "a list comprehension"
#filtered_sentence = [w for w in word_tokens if not w in stop_words]

#define a list to hold test whose stop words have been removed
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words:
        #stemword is user defined function
        #Stemword takes words back to their stem/root
        filtered_sentence.append(stemword(w))

text_file=' '.join(filtered_sentence)# convert from list to string
text_file=text_file.replace('#','\n\n')# Reconstruct sentences but without stopwords and fullstops
#progess communication: preprocessed topic file ready for test features generation
print('File: '+in_file+' preprocessed')
#obtains the path where the file will be saved
pp_out_path=easygui.filesavebox(msg='Confirm or change details of your file', title='Save Preprocessed File', default=in_file+'.txt', filetypes=' \*.txt')
#saves the file in the provided path
with open(pp_out_path, 'w',encoding='UTF8') as f:
    f.write(text_file)
f.close()
print('Preprocessed Topics file: '+in_file+' Saved in '+pp_out_path)#dispalys to the user the location of the saved file
#Preprocessed topics written in different files for ease of reference during features generation
with open(pp_out_path, 'r',encoding='UTF8') as fp:
    line =fp.readline()
    cnt = 1
    fname=''#name of the file storing the sentence
    str1=''#holder of the single sentence
    while line:
        pp_out_path_ft = pp_out_path #path to individual topic files| ft=file topic
        str1=format(line.strip())# removal of leading spaces
        if(str1 != ''):#ignore empty sentences
            if str1[0:2]=='d0':#detect a string that is a topic label
                fname=str1[0:6].upper()#make the topic label the file name
                line=fp.readline()
                line=fp.readline()
                line=fp.readline()
            #process topic narration and store it in the topic file . this wil be the cluster query
            pp_out_path_ft=pp_out_path_ft[0:27]+'\\'+fname+ '.txt'
            with open(pp_out_path_ft, 'a',encoding='UTF8') as ft:
                ft.write(line + "\n")#write clsuter query in the topic file
            ft.close()
        line=fp.readline()
        #cnt+=1
fp.close()
#displays to the user the location of the saved individual topic file
print('Preprocessed individual topic: '+fname+' Saved in '+pp_out_path_ft)

#End of FIle

