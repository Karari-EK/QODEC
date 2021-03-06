# QODEC
For Query Oriented Deep Extraction Enhanced with Classification (QODEC)
All files provided herein are written, tested and used in research on Query Oriented Deep Extraction enhanced with Classification (QODEC) 
QODEC is part of my PhD work in which hybridization of unsupervised deep learning and supervised learning techniques was studied to establish chances of enhancing multi-document extractive summarization.
The two dataset used in this research are DUC 2006 and DUC 2007, both from the National Institute of standards and Technology (NIST), Document Understanding Conference (DUC), United States of America. DUC 2006 dataset is used in it’s entirely for training; that is, preparing the labelled data while DUC 2007 is used in testing.
All files in this repository are source code, dataset, preprocessed for NLP, preprocessed for SVM or DBN input matrix/vector files. 
The naming scheme for project code files is made of four characters which are short form of the actual file description
Files 1 to 11 described, below are mainly used for preprocessing for NLP and for SVM and DBN input
1.	#Text File Conversion to CSV code (csvc1.0.0) #converts .txt files into csv files en mass, # i.e. all files in the 50 folders in one run #conversion prepares for easier data manipulation and viewing
2.	#Document Preparation and Preprocessing Code (dppc1.0.0) #Document Preparation converts HTML code to Text #Preprocessing prepares text for NLP
3.	#Derive Summary Code (dsumc1.0.0) #determine zero or 1 based on certain sentence features, tf, ss and sp
4.	#Features Generation Code (fegc1.0.0) #Generates the features matrix given the preprocessed text (document & topic) #Where necessary conducts further preprocessing before features generation #Features generated include : #Term Frequency(tf), Significant Factor(sf),Temporal Dimension(td) #and sentence length penalty (in this case a reward to non-verbose sentences)
5.	#Extract estimate summary code (eesuc1.0.0) #Aggregate the 0 or 1 generated by users
6.	#Extract User Summaries Code(eusc1.0.0) #Extract zeroes and ones from user summary files/focus group files #Extracted summary per user stored in a text file which is the transferred to an excel file
7.	#Extract features from features.csv code (efcc1.0.0) #Extract all features and the summary column as well from featurs.csv files # Aggregates all features into one big matrix (effectively creating the labeled data) in readiness for SVM input 
8.	#Serialization of test files code (stfc1.0.0) #Generates serial numbers for test files  #Serialization easies generation of training data for supervised training #Serialization enables tracing of sentences in formation of summaries
9.	#Topics Preparation and Preprocessing Code (tppc1.0.0) #Topics Preparation converts HTML code to Text #Preprocessing prepares text for NLP
10.	#Summary Generation Code (sugc1.0.0) #Generates a summary from flagged text files #can be modified to have the summary written in a text file
11.	#Serialization of preprocessed files code (spfc1.0.0) #Generates serial numbers for sentences per file for all preprocessed files #Used to compare with test files serialization #Comparison confirms referential integrity/ no of sentences is maintained after preprocessing
Files 12 to 15 shown, below are mainly used for actual testing of the hybrid algorithm
1.	#Support Vector Machine Code (svmc1.0.0) #generates a summary by classification based on pre-training #input includes: #Training features matrix with user summary #Data for which classification/prediction will be done #data for prediction is basically a matrix without the user summary column #the summary column will be the prediction
2.	#Deep Belief Network Code(DBN) (dbnc1.0.0) #This DBN is based on Restricted Boltzmann Machine (RBM) #generates a summary by deep learning #input includes: #Training features matrix consisting of features sets that have 1 in the user summary #the user summary is itself however excluded in input set.#Data for which classification/prediction will be done #data for prediction is basically a matrix without the user summary column #the summary column will be the prediction
3.	#Deep Belief Network-Support Vector Machine (DBN-SVM)hybrid Code (svmc1.0.0) #generates a summary post-classification hybridization of SVM #input includes: #Output from the Deep Belief Network
4.	#Support Vector machine-Deep Belief Network Code (svdc1.0.0) #This SVM-DBN is based on Restricted Boltzmann Machine (RBM) #generates a summary by deep learning #input includes: #Training features matrix consisting of features sets that have 1 in the user summary #the user summary is itself however excluded in input set. #Data for which classification/prediction will be done #data for prediction is basically a matrix without the user summary column# the summary column will be the prediction
All data (Both DUC 2007 and DUC 2006) is available on request from NIST –DUC. All content herein is also available from Github but on request. 
Contact efantus.kinyanjui@dkut.ac.ke or efantusk@gmail.com  for more if need be.
