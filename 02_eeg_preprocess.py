# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:46:07 2021

@author: INFINITE-WORKSTATION
"""
import mne

#Just display errors and warnings
mne.set_log_level("WARNING")

#Loading EEG data
data = mne.io.read_raw_gdf("motorimagination_subject1_run1.gdf", preload=True) 

eeg = data.get_data()

s_freq = 512
eeg_chl = ['F3','F1','Fz','F2','F4','FFC5h','FFC3h','FFC1h','FFC2h','FFC4h',
           'FFC6h','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FTT7h','FCC5h',
           'FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h','C5','C3','C1','Cz',
           'C2','C4','C6','TTP7h','CCP5h','CCP3h','CCP1h','CCP2h','CCP4h',
           'CCP6h','TTP8h','CP5','CP3','CP1','CPz','CP2','CP4','CP6','CPP5h',
           'CPP3h','CPP1h','CPP2h','CPP4h','CPP6h','P3','P1','Pz','P2','P4',
           'PPO1h','PPO2h','eog-l','eog-m','eog-r']

info = mne.create_info(eeg_chl, s_freq, ch_types=["eeg"] * 61 + ["eog"]*3)
raw = mne.io.RawArray(eeg[0:64,:], info)
raw.set_montage("standard_1005")
raw.set_eeg_reference('average', projection=True)

raw_tmp = raw.copy()
raw_tmp.filter(1,40, fir_design='firwin', skip_by_annotation='edge')

#####################  Calling ICA ###################################

ica = mne.preprocessing.ICA(method="infomax",
                            fit_params={"extended": True},
                            random_state=1)

#######################################################################
#################### ploting PSD ######################################

raw.plot_psd(area_mode='range', tmax=10.0, average=False)
raw_tmp.plot_psd(area_mode='range', tmax=10.0)

########################################################################
################### ICA Analysis #######################################

ica.fit(raw_tmp)
ica.plot_sources(inst=raw_tmp)
ica.plot_components(inst=raw_tmp, picks=None) #to pick all channels - none
ica.exclude = [11]

raw_corrected = raw_tmp.copy()
ica.apply(raw_corrected)

#########################################################################
################## Ploting filttred and Pre-prcossed  EEG ###############

raw.plot(n_channels=64, start=0.0, duration=10, scalings=dict(eeg=500e-6))
raw_corrected.plot(n_channels=64, start=0.0, duration=10,scalings=dict(eeg=500e-6))

eeg_preprocess = raw_corrected.get_data()

savemat("s01_crest_eeg_preprocess.mat",{"classrest_eeg_proc":eeg_preprocess,
                                      "EEG_chl":eeg_chl})
# raw_corrected.filter(0.3,12, fir_design='firwin', skip_by_annotation='edge')
