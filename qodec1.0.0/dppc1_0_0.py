#Document Preparation and Preprocessing Code (dppc1.0.0)
#Document Preparation converts HTML code to Text
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
import os
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


##functions defination section

#function stemword
def stemword(w):
    #invoke the stemmer.stem and return the stemmed word 
    return (stemmer.stem(w))

#function create_document
#Extracted file converted so that each sentence
#is in its own paragraph sparated by sace
def create_document(t_file):
    t_file=t_file.replace('.\n','#')#mark sentences that are also end of paragraphs 
    t_file=t_file.replace('\n',' ')#remove unwanted/illegal paragraphs
    t_file=t_file.replace('. ','#')#mark remaining sentences
    t_file=t_file.replace('#','.\n\n')#create a paragraph for each sentence
    t_file=t_file.replace('\n ','\n')#remove leading spaces in every paragraph
    return (t_file)

#function main_preprocessing(text_file)
#converts to lowercase
#removes unwanted characters
#removes stop words
#tokenize the sentence i readness for stop word removal
#remove stop words
#take words to their root/stem ---invokes stemword function
#reconstruct the sentence but without stopwords and fullstops
def main_preprocessing(t_file):
    #change end of sentence character to #
    #Enables reconstruction of sentences after stop words removal
    t_file=t_file.replace('.\n\n','#')
    t_file = re.sub('[^a-zA-Z0-9\n#]', ' ', t_file)#remove special and unwanted characters
    t_file=t_file.lower()#convert all text to lower case; facilitates stop word removal

    #remove stop words and converts words to their stem/root
    stop_words = set(stopwords.words('english'))# define stopwords object 
    word_tokens = word_tokenize(t_file)#split text to words, word_tokens is a list
    #removes all stopwords from word tokens, filtered_sentence is a list

    #Alternative code for removal of stopwords using "a list comprehension"
    ##filtered_sentence = [w for w in word_tokens if not w in stop_words]

    #define a list to hold text whose stop words have been removed
    filtered_sentence = [] 
      
    for w in word_tokens: 
        if w not in stop_words:
            #stemword is user defined function
            #Stemword takes words back to their stem/root
            filtered_sentence.append(stemword(w))

    t_file=' '.join(filtered_sentence)# convert from list to string
    t_file=t_file.replace('#','\n\n')# Reconstruct sentences but without stopwords and fullstops
    return t_file

##End of function defination section


#select the input file
print('Enter the file to be preprocessed')
in_path=easygui.fileopenbox()#define path of the input file (html or .0123 type)

#Provide for iteration through the files
root_path=in_path
rootdir = in_path[0:20]
#print('Root Folder: '+rootdir)#'C:\pfiles\Test Data\'
dir_num=0#folder number
total_files=1#overall file count

#for subdir, dirs, files in os.walk(rootdir):
    #rootdir=subdir
    #print('Folder N0 : '+ str(dir_num)+' '+subdir)
    #dir_num+=1

for subdir, dirs, files in os.walk(rootdir):
    #folder=in_path[20:26]
    file_num=1#file number for files in a single directory
    for file in files:
        in_path=os.path.join(subdir, file)
        folder=in_path[20:26]
        print(folder)
        in_file=in_path[27:38]#extract file name from file path
        print('File N0:'+ str(file_num)+' of overall count: '+str(total_files)+' named: '+in_file+' submitted for processing')
            
            
        #remove HTML code
        html_file = open(in_path, 'r')#open html file
        source_code = html_file.read()#read html file
        text_file=h.handle(source_code)#remove all html code 

        #convert text to sentences
        text_file=create_document(text_file)

        #progress communication: Content for generating human summary ready
        print('File: '+in_file+' Processed for generation of human summary')

        #obtains the path where the file will be saved
        #print('Select the location where the html extracted file will be saved')
        #out_path=easygui.filesavebox(msg='Confirm or change details of your file', title='Save Document', default=in_file+'.txt', filetypes=' \*.txt')
        print("Folder...." + folder)
        out_path='C:\\pfiles\\Test Files\\'+folder+'\\'+in_file+'.txt'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)#create folder that does not exist
        #saves the file in the provided path
        with open(out_path, 'w',encoding='UTF8') as f:
            f.write(text_file)
        print('Document file: '+in_file+' Saved in '+out_path)#dispalys to the user the location of the saved file
        
        #Text Preprocessing
        print('File :'+in_file+' submitted for preprocessing')
        text_file=main_preprocessing(text_file)
        #progess communication: preprocessed files ready for text features generation
        print('File: '+in_file+' preprocessed')

        #obtains the path where the file will be saved
        #print('Select the location where the preprocessed file will be saved')
        #pp_out_path=easygui.filesavebox(msg='Confirm or change details of your file', title='Save Preprocessed File', default=in_file+'.txt', filetypes=' \*.txt')
        pp_out_path='C:\\pfiles\\PPFiles\\'+folder+'\\'+in_file+'.txt'
        os.makedirs(os.path.dirname(pp_out_path), exist_ok=True)#create folder that does not exist
        #saves the file in the provided path
        with open(pp_out_path, 'w',encoding='UTF8') as f:
            f.write(text_file)
        print('Preprocessed file: '+in_file+' Saved in '+pp_out_path)#dispalys to the user the location of the saved file
        
        #increament file counters
        file_num+=1 #number of files per folder
        total_files+=1 #overall number of files
        
#End of FIle
