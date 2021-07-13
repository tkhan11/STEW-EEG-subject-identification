#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
## Version history:

2021, May:
	Author: Tanveer Khan
"""

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from scipy.signal import butter,filtfilt,find_peaks,find_peaks, resample
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve,auc, precision_score,recall_score,f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from keras import layers, models, regularizers
import mne
from mne import find_events, fit_dipole
from autoreject import AutoReject
import seaborn as sns
import sys
import statistics as st
from keras.utils import to_categorical
#import pywt
import sys
import antropy as an 
import time

import warnings
warnings.filterwarnings("ignore")


Channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]   

##1.1 Filter requirements.
T =     150         # Sample Period,    Seconds
fs =    128         # Sample rate,      Hz
cutoff = 40          # Desired cutoff frequency of the filter, Hz
nyq = 0.5 * fs      # Nyquist Frequency
order = 1           # sin wave can be approx represented as quadratic
n = int(T * fs)+1   # Total number of samples

t=np.linspace(0,150,19200)


"""

************* ONE TIME EXECUTION OF THIS SECTION CODE TO PREPARE FEATURES ***************


Features = ["mean_PSD",         "STD_PSD",
            "A_mean",           "A_STD",            "A_Var",
            "A_range",          "A_skew",           "A_kurtosis",
            "Permutation_E",    "Spectral_E",       "SVD_E",
            "Approximate_E",    "Sample_E",         "Petrosian_FD",
            "Katz_FD",          "Higuchi_FD",       
            "Detrended fluctuation analysis",
            "Label"] 
                              # Features' Names

Features_df = pd.DataFrame(columns = Features)    # Features of Subject 1

normal_cutoff = [3/nyq ,40/nyq]
b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)    # Get the filter coefficients
SS=0
TIMES = np.zeros([48])

# Read Data
start_time = time.time()
for s in range(1,49):
    if s < 10:
        URL=".\\STEW Dataset\\sub0"+str(s)+"_hi.txt"
    else:
        URL=".\\STEW Dataset\\sub"+str(s)+"_hi.txt"

    Data = pd.read_csv(URL, sep="  ", header=None)
    Data.columns=Channels
    print("Extracting Features From Subject: ",s)
    print("--- %s seconds ---" % (time.time() - start_time))
    TIMES[s-1] = time.time() - start_time

    for Ch in Channels:
        # Pre-Processing
        Data.insert(0, ''.join([Ch + " Filtered"]), filtfilt(b,a,Data[Ch]))
        Data = Data.drop(Ch, axis=1)
        Data.insert(0, ''.join([Ch + " Despiked"]), Data[''.join([Ch + " Filtered"])].where(Data[''.join([Ch + " Filtered"])] < Data[''.join([Ch + " Filtered"])].quantile(0.97), Data[''.join([Ch + " Filtered"])].mean()))
        Data = Data.drop(''.join([Ch + " Filtered"]), axis=1)
        Data.insert(0, Ch, Data[''.join([Ch + " Despiked"])].where(Data[''.join([Ch + " Despiked"])] > Data[''.join([Ch + " Despiked"])].quantile(0.05), Data[''.join([Ch + " Despiked"])].mean()))
        Data = Data.drop(''.join([Ch + " Despiked"]), axis=1)
    w=0
    for window in range(0,60):
        EEG_Window = Data[w:w+640] # Windowing, Window Len: 5 Sec, Overlap: 2.5 Sec
        w=w+320
        Fet = np.zeros([300]) #Temporal Features + Lable array
        Fet_ch = np.zeros([18])
        i=0
        for i in range(14):
            channel = EEG_Window[Channels[i]]
            # Feature Extraction
   

            # PSD Features
            f, Pxx =scipy.signal.welch(channel,fs) #Extract PSD according to Welch thiorem

            Fet[i]         = mean(Pxx)                                                      # Mean of PSD
        
            Fet[i+14]       = std(Pxx)                                                      # Standered Deviation of PSD
        
            # Statistics Features
            Fet[i+28]      = mean(channel)                                                  # Amplitude Mean
        
            Fet[i+42]      = std(channel)                                                   # Amplitude Standered Deviation
            
            Fet[i+56]      = np.var(channel)                                                # Amplitude variance
            
            Fet[i+70]      = max(channel)-min(channel)                                      # Amplitude Range
            
            Fet[i+84]      = skew(channel)                                                  # Amplitude Skew
            
            Fet[i+98]      = kurtosis(channel)                                              # Amplitude kurtosis
            
            # Entropy Features
            Fet[i+112]      = an.perm_entropy(channel, order=3, normalize=True)                 # Permutation entropy
            
            Fet[i+126]      = an.spectral_entropy(channel, 100, method='welch', normalize=True) # Spectral entropy
            
            Fet[i+140]      = an.svd_entropy(channel, order=3, delay=1, normalize=True)         # Singular value decomposition entropy
            
            Fet[i+154]      = an.app_entropy(channel, order=2, metric='chebyshev')              # Approximate entropy
            
            Fet[i+168]      = an.sample_entropy(channel, order=2, metric='chebyshev')           # Sample entropy
            
            # Fractal dimension Features
            Fet[i+182]      = an.petrosian_fd(channel)                                          # Petrosian fractal dimension
                
            Fet[i+196]      = an.katz_fd(channel)                                               # Katz fractal dimension
            
            Fet[i+210]      = an.higuchi_fd(channel, kmax=10)                                   # Higuchi fractal dimension
            
            Fet[i+224]      = an.detrended_fluctuation(channel)                                 # Detrended fluctuation analysis
            
        Fet_ch[0]      = mean(Fet[0:14])        # Mean of PSD
        Fet_ch[1]      = mean(Fet[14:28])       # Standered Deviation of PSD
        Fet_ch[2]      = mean(Fet[28:42])       # Amplitude Mean
        Fet_ch[3]      = mean(Fet[42:56])       # Amplitude Standered Deviation
        Fet_ch[4]      = mean(Fet[56:70])       # Amplitude variance
        Fet_ch[5]      = mean(Fet[70:84])       # Amplitude Range
        Fet_ch[6]      = mean(Fet[84:98])       # Amplitude Skew
        Fet_ch[7]      = mean(Fet[98:112])      # Amplitude kurtosis
        Fet_ch[8]      = mean(Fet[112:126])     # Permutation entropy
        Fet_ch[9]      = mean(Fet[126:140])     # Spectral entropy
        Fet_ch[10]     = mean(Fet[140:154])     # Singular value decomposition entropy
        Fet_ch[11]     = mean(Fet[154:168])     # Approximate entropy
        Fet_ch[12]     = mean(Fet[168:182])     # Sample entropy
        Fet_ch[13]     = mean(Fet[182:196])     # Petrosian fractal dimension
        Fet_ch[14]     = mean(Fet[196:210])     # Katz fractal dimension
        Fet_ch[15]     = mean(Fet[210:224])     # Higuchi fractal dimension
        Fet_ch[16]     = mean(Fet[224:238])     # Detrended fluctuation analysis
        Fet_ch[17]     = s-1  
        
        Features_df.loc[SS]=Fet_ch
        SS=SS+1

'''
for i in range(47):
    TIMES[i] = TIMES[i+1]-TIMES[i]
print("Mean Feature Extraction Time: ", mean(TIMES[0:47]))

'''

Features_df.to_csv('extractedFeatures.csv')

"""


print("All features were extracted and saved")

print("Begining Data Preparation")

Features_df = pd.read_csv('./extractedFeatures.csv')
Features_df.drop(Features_df.columns[0], axis = 1, inplace = True)      # In the csv file first column has number values from 0-2280 and carries redundadnt info.

#3 Data preparation
##3.1 Shuffling
Data = Features_df.sample(frac = 1) 
features = Data[[x for x in Data.columns if x not in ["Label"]]]   # Data for training
Labels = Data['Label']                                            # Labels for training
Labels = Labels.astype('category')

'''

*************** THIS SECTION HAS GRAPH PLOTTING CODE ************************

#4 Feature Selection 

##4.1 Correlation Map
#print("Plotting Correlation Map\n")

# Correlation Matrix
Corelation_df = features.corr()

# Continuous Visualization
plt.imshow(Corelation_df, cmap='hot', interpolation='nearest')
#plt.show()


# Discrete Visualization
sns.set(font_scale=0.9)
ax = sns.heatmap(Corelation_df, linewidth=0.5,annot=True)
#plt.show()

sns.clustermap(data=Corelation_df, annot=True,linewidth=1,cmap = "Accent",annot_kws={"size": 8},)
#plt.show()

# Scatter Plots
## Warning: Operating the next line will take a lot of time
#sns.pairplot(Corelation_df)

print("Corelation map plot done")

'''

'''
This code having an issue in "Labels" attribute
print("Show Corelation with label")

features_Corelation = Corelation_df["Label"]
features_Corelation = features_Corelation[:-1]

features_Corelation_Ordered=abs(features_Corelation.sort_values(ascending =False))
print(features_Corelation_Ordered)

'''

'''
#4.2 Univariate Selection
#apply SelectKBest class to extract top 17 best features

print("Start Univariate Selection" )

bestfeatures = SelectKBest(score_func=chi2, k=17)
fit = bestfeatures.fit(abs(features),Labels)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores = featureScores.set_index('Specs')
featureScore_Ordered = featureScores.sort_values(by = "Score", ascending =False)
#print(featureScores.nlargest(17,'Score'))  #print 17 best features



print("End Univariate Selection" )
print("Start Feature Importance" )



##4.3 Feature Importance
model = ExtraTreesClassifier()
model.fit(features,Labels)

#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances_Ordered = feat_importances.sort_values(ascending =False)
print(feat_importances_Ordered)




# Plot Bar char for each score
print("Plot Bar char for each score")

#features_Correlation_Ordered.nlargest(17).plot(kind='bar')
#plt.show()

featureScore_Ordered.nlargest(17,"Score").plot(kind='bar')
plt.show()

feat_importances_Ordered.nlargest(17).plot(kind='bar')
plt.show()


# Compare methods
#Featuers_df = pd.concat([features_Correlation,featureScores, feat_importances], axis=1)
#Featuers_df.columns = ["Correlation","Score"," Importance"]
Features_df = pd.concat([featureScores, feat_importances], axis=1)
Features_df.columns = ["Score"," Importance"]


x = Features_df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
Featuers_df = pd.DataFrame(x_scaled)

Scores_df = pd.concat([ pd.DataFrame(Features), Features_df], axis=1)
#Scores_df.columns = ["Features","Correlation","Score","Importance"]
Scores_df.columns = ["Features","Score","Importance"]
Scores_df=Scores_df[:-1]
Scores_df=Scores_df.set_index("Features")

Features_df = pd.concat([ pd.DataFrame(Features), Features_df], axis=1)
#Features_df.columns = ["Features","Correlation","Score","Importance"]
Features_df.columns = ["Features","Score","Importance"]
Features_df=Features_df.set_index("Features")
Features_df=Features_df[:-1]


print("Without Normalization")
print(Features_df)

ay = sns.heatmap(Features_df, linewidth=0.5,annot=True)
plt.show()


print("With Normalization")
print(Scores_df)
ax = sns.heatmap(Scores_df, linewidth=0.5,annot=True)
plt.show() 



# Feature Selection
Reduced_Features = [ "A_Var",
            "A_kurtosis",
            "Permutation_E",           
            "Detrended fluctuation analysis",
            "Label"] 

Reduced_Data = Data[Reduced_Features]

'''

##3.3 Prepare Train and test Data
splitRatio = 0.3
train, test = train_test_split(Data ,test_size=splitRatio,
                               random_state = 123, shuffle = True)  # Spilt to training and testing data 

train_X = train[[x for x in train.columns if x not in ["Label"]]]   # Data for training
train_Y = train['Label']                                            # Labels for training

###4.5.2 Testing Data
test_X = test[[x for x in test.columns if x not in ["Label"]]]     # Data fo testing
test_Y = test["Label"]                                              # Labels for training

###4.5.3 Validation Data
x_val = train_X[:200]                                                # 50 Sample for Validation
partial_x_train = train_X[200:]
partial_x_train = partial_x_train.values

y_val = train_Y[:200]
y_val = to_categorical(y_val)
partial_y_train = train_Y[200:]
partial_y_train = partial_y_train.values
partial_y_train = to_categorical(partial_y_train)

print("Data is prepeared")

print("Start Building Classifer")

#4 Classification Model

##4.1 Building Model

###4.1.1 Architecture
model = models.Sequential()
model.add(layers.Dense(200, activation = 'relu', input_shape=(17,),kernel_regularizer=regularizers.l1(0.01)))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(150, activation = 'relu'))
model.add(layers.Dense(100, activation = 'relu'))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(75, activation = 'relu'))
model.add(layers.Dense(48,  activation= 'softmax'))

####5.1.1.2 Hyper Parameters Tuning
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Classifier Bulit\n")
print("Start Training\n")

##5.1.2 Training Model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 5,                # initial value = 1000
                    batch_size = 16,              # initial value = 16
                    validation_data=(x_val, y_val))

weights = model.get_weights() 
configs = model.get_config() 

Featuers_weights = np.apply_along_axis(mean, 1, weights[0])



print("Finish Training")


#5 Model Evaluation

##5.1 Network Architecture
print(model.summary())

print("Start Evaluating Data")

##4.2 Training Process
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)
plt.figure()
plt.plot(epochs, loss_values, 'bo', label="training loss", color='r')
plt.plot(epochs, val_loss_values, 'b', label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoches")
plt.ylabel("loss")
plt.legend()
plt.show()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.figure()
plt.plot(epochs, acc_values, 'bo', label="training acc", color = 'r')
plt.plot(epochs, val_acc_values, 'b', label="validation acc")
plt.title("Training and Validation acc")
plt.xlabel("Epoches")
plt.ylabel("acc")
plt.legend()
plt.show()

##5.3 Prediction
ANN_predictions = model.predict(test_X)

Pred = np.zeros([len(ANN_predictions)])
for i in range(0,len(ANN_predictions)):
    Pred[i] = list(ANN_predictions[i]).index(max(ANN_predictions[i]))
ANN_Pred = pd.Series(Pred)

####5.1.2.4 Metrics
print("Accuracy:",accuracy_score(test_Y, ANN_Pred))
print("f1 score:", f1_score(test_Y, ANN_Pred,average="micro"))
print("precision score:", precision_score(test_Y, ANN_Pred,average="micro"))
print("recall score:", recall_score(test_Y, ANN_Pred,average="micro"))
print("confusion matrix:\n",confusion_matrix(test_Y, ANN_Pred))
print("classification report:\n", classification_report(test_Y, ANN_Pred))

'''
results=pd.DataFrame({"Accuracy":accuracy_score(test_Y, ANN_Pred),
                      "f1 score":f1_score(test_Y, ANN_Pred,average="micro"),
                      "precision score":precision_score(test_Y, ANN_Pred,average="micro"),
                      "recall score":recall_score(test_Y, ANN_Pred,average="micro"})
                     
results.to_csv("results.csv")
'''
##5.6 Plots

####5.1.2.5 Plots

# plot Confusion Matrix as heat map
plt.figure(figsize=(3,2))
sns.heatmap(confusion_matrix(test_Y, ANN_Pred),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)
plt.show()

###5.6.2 plot ROC curve
##test_Y_01 = test_Y_01.cat.codes

#fpr,tpr,thresholds = roc_curve(test_Y, ANN_Pred)
#plt.subplot(222)
#plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
#plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
#plt.legend(loc = "best")
#plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=15)

