## Separate dataset to 9000 samples (30 recording for each class )

import wfdb

#Read data from file 
record = wfdb.rdrecord   #Read ECG data
annotations = wfdb.rdann #Read atr
#
ecg_signal = record.p_signal    #