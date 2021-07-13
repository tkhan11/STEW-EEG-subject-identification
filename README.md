# SIMULTANEOUS TASK EEG WORKLOAD DATASET -subjects’ identification (STEW-SI)

**STEW-SI** is a lightweight artificial neural network based subject’s identification framework implemented in Python with categorical feature support**.

**About STEW dataset:**
This dataset consists of raw EEG data from 48 subjects who participated in a multitasking workload experiment utilizing the SIMKAP multitasking test. The subjects’ brain activity at rest was also recorded before the test and is included as well. The Emotiv EPOC device, with sampling frequency of 128Hz and 14 channels was used to obtain the data, with 2.5 minutes of EEG recording for each case. Subjects were also asked to rate their perceived mental workload after each stage on a rating scale of 1 to 9 and the ratings are provided in a separate file.

**Instructions:** 
The data for each subject follows the naming convention: subno_task.txt. For example, sub01_lo.txt would be raw EEG data for subject 1 at rest, while sub23_hi.txt would be raw EEG data for subject 23 during the multitasking test. The rows of each datafile corresponds to the samples in the recording and the columns corresponds to the 14 channels of the EEG device: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4, respectively.

**Methodology:**
From the pre-processed EEG signal 17 statistical, entropy, and energy features were extracted. For performing the identification task Artificial neural network (ANN) was used. 

### Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

