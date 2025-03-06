import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy.io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(150)
data_path = 'data_path'
data = pd.read_csv(data_path)
data = data.iloc[:,1:]
participants = list(data['Participant'].unique())
print(len(data['Recording'].unique()))
print('Total Participants involved in this study {}'.format(len(participants)))

######################## Epochs Function ####################
def epoch_data(data,window = None, train_dict = None, test_dict  = None):
  train_dict = {key: [filename.replace('_RD', '') for filename in value] for key, value in train_dict.items()}
  test_dict = {key: [filename.replace('_RD', '') for filename in value] for key, value in test_dict.items()}

  participants = list(data['Participant'].unique())
  X_train = []
  Y_train = []
  X_test  = []
  Y_test  = []
  counter = 0
  for part in participants:

    part_data = data[data['Participant'] == part]
    recordings = list(part_data['Recording'].unique())
    #print('Working on Participant {}'.format(part))
    for rec in recordings:
      dat = data[(data['Participant']== part) & (data['Recording'] == rec)]['Signal']
      #print('Working on Recording {} with shape {}'.format(rec, dat.shape))
      dat_arr = np.array(dat).reshape(-1,1)

      if window is not None:
        for i in range(0,dat_arr.shape[0],window):
          if i + window < dat_arr.shape[0]:
            if part in train_dict.keys():

              if rec in train_dict[part]:
                X_train.append([dat_arr[i:i+window,:]])
                Y_train.append(part)
            if part in test_dict.keys():
              if rec in test_dict[part]:
                X_test.append([dat_arr[i:i+window,:]])
                Y_test.append(part)        
      else:
        flag_test   = False
        flag_train  = False
        if part in train_dict.keys():
          if rec in train_dict[part]:
            X_train.append([dat_arr])
            Y_train.append(part)
            flag_train = True
        if part in test_dict.keys():
          if rec in test_dict[part]: 
            X_test.append([dat_arr])
            Y_test.append(part)
            flag_test = True
        if flag_test  == False:
          if flag_train == False:
            print(part)
            print(rec)

    counter += 1
    print('Done Participant {}'.format(counter))

  return X_train,Y_train, X_test, Y_test








train_dict = {'Person_015': ['m015_38_wksp_RD.mat', 'm015_32_wksp_RD.mat', 'm015_40_wksp_RD.mat', 'm015_8_wksp_RD.mat', 'm015_6_wksp_RD.mat', 'm015_36_wksp_RD.mat', 'm015_46_wksp_RD.mat', 'm015_48_wksp_RD.mat', 'm015_27_wksp_RD.mat', 'm015_17_wksp_RD.mat', 'm015_35_wksp_RD.mat', 'm015_44_wksp_RD.mat', 'm015_4_wksp_RD.mat', 'm015_3_wksp_RD.mat', 'm015_30_wksp_RD.mat', 'm015_25_wksp_RD.mat', 'm015_34_wksp_RD.mat', 'm015_16_wksp_RD.mat', 'm015_50_wksp_RD.mat', 'm015_24_wksp_RD.mat', 'm015_37_wksp_RD.mat', 'm015_28_wksp_RD.mat', 'm015_23_wksp_RD.mat', 'm015_7_wksp_RD.mat', 'm015_2_wksp_RD.mat', 'm015_21_wksp_RD.mat', 'm015_19_wksp_RD.mat', 'm015_13_wksp_RD.mat', 'm015_15_wksp_RD.mat', 'm015_14_wksp_RD.mat', 'm015_39_wksp_RD.mat', 'm015_20_wksp_RD.mat', 'm015_49_wksp_RD.mat', 'm015_10_wksp_RD.mat', 'm015_45_wksp_RD.mat'], 'Person_009': ['m009_45_wksp_RD.mat', 'm009_4_wksp_RD.mat', 'm009_44_wksp_RD.mat', 'm009_31_wksp_RD.mat', 'm009_32_wksp_RD.mat', 'm009_14_wksp_RD.mat', 'm009_40_wksp_RD.mat', 'm009_29_wksp_RD.mat', 'm009_10_wksp_RD.mat', 'm009_33_wksp_RD.mat', 'm009_25_wksp_RD.mat', 'm009_3_wksp_RD.mat', 'm009_8_wksp_RD.mat', 'm009_27_wksp_RD.mat', 'm009_50_wksp_RD.mat', 'm009_35_wksp_RD.mat', 'm009_36_wksp_RD.mat', 'm009_18_wksp_RD.mat', 'm009_34_wksp_RD.mat', 'm009_26_wksp_RD.mat', 'm009_2_wksp_RD.mat', 'm009_15_wksp_RD.mat', 'm009_37_wksp_RD.mat', 'm009_23_wksp_RD.mat', 'm009_47_wksp_RD.mat', 'm009_24_wksp_RD.mat', 'm009_28_wksp_RD.mat'], 'Person_007': ['m007_22_wksp_RD.mat', 'm007_35_wksp_RD.mat', 'm007_14_wksp_RD.mat', 'm007_37_wksp_RD.mat', 'm007_6_wksp_RD.mat', 'm007_45_wksp_RD.mat', 'm007_23_wksp_RD.mat', 'm007_43_wksp_RD.mat', 'm007_3_wksp_RD.mat', 'm007_1_wksp_RD.mat', 'm007_44_wksp_RD.mat', 'm007_24_wksp_RD.mat', 'm007_38_wksp_RD.mat', 'm007_10_wksp_RD.mat', 'm007_4_wksp_RD.mat', 'm007_7_wksp_RD.mat', 'm007_48_wksp_RD.mat', 'm007_32_wksp_RD.mat', 'm007_31_wksp_RD.mat', 'm007_9_wksp_RD.mat', 'm007_8_wksp_RD.mat', 'm007_33_wksp_RD.mat', 'm007_42_wksp_RD.mat', 'm007_28_wksp_RD.mat', 'm007_30_wksp_RD.mat', 'm007_47_wksp_RD.mat', 'm007_2_wksp_RD.mat', 'm007_27_wksp_RD.mat', 'm007_20_wksp_RD.mat', 'm007_40_wksp_RD.mat', 'm007_5_wksp_RD.mat'], 'Person_008': ['m008_16_wksp_RD.mat', 'm008_10_wksp_RD.mat', 'm008_12_wksp_RD.mat', 'm008_24_wksp_RD.mat', 'm008_1_wksp_RD.mat', 'm008_49_wksp_RD.mat', 'm008_28_wksp_RD.mat', 'm008_42_wksp_RD.mat', 'm008_33_wksp_RD.mat', 'm008_20_wksp_RD.mat', 'm008_13_wksp_RD.mat', 'm008_47_wksp_RD.mat', 'm008_7_wksp_RD.mat', 'm008_23_wksp_RD.mat', 'm008_44_wksp_RD.mat', 'm008_22_wksp_RD.mat', 'm008_30_wksp_RD.mat', 'm008_32_wksp_RD.mat', 'm008_25_wksp_RD.mat', 'm008_34_wksp_RD.mat', 'm008_18_wksp_RD.mat', 'm008_31_wksp_RD.mat', 'm008_46_wksp_RD.mat', 'm008_17_wksp_RD.mat', 'm008_5_wksp_RD.mat', 'm008_2_wksp_RD.mat', 'm008_3_wksp_RD.mat', 'm008_48_wksp_RD.mat', 'm008_4_wksp_RD.mat', 'm008_9_wksp_RD.mat', 'm008_15_wksp_RD.mat', 'm008_38_wksp_RD.mat', 'm008_14_wksp_RD.mat'], 'Person_012': ['m012_41_wksp_RD.mat', 'm012_17_wksp_RD.mat', 'm012_40_wksp_RD.mat', 'm012_42_wksp_RD.mat', 'm012_4_wksp_RD.mat', 'm012_32_wksp_RD.mat', 'm012_37_wksp_RD.mat', 'm012_43_wksp_RD.mat', 'm012_20_wksp_RD.mat', 'm012_30_wksp_RD.mat', 'm012_18_wksp_RD.mat', 'm012_11_wksp_RD.mat', 'm012_3_wksp_RD.mat', 'm012_13_wksp_RD.mat', 'm012_44_wksp_RD.mat', 'm012_15_wksp_RD.mat', 'm012_7_wksp_RD.mat', 'm012_34_wksp_RD.mat', 'm012_31_wksp_RD.mat', 'm012_21_wksp_RD.mat', 'm012_28_wksp_RD.mat', 'm012_5_wksp_RD.mat', 'm012_23_wksp_RD.mat', 'm012_19_wksp_RD.mat', 'm012_9_wksp_RD.mat', 'm012_26_wksp_RD.mat', 'm012_47_wksp_RD.mat', 'm012_6_wksp_RD.mat', 'm012_49_wksp_RD.mat', 'm012_36_wksp_RD.mat', 'm012_50_wksp_RD.mat', 'm012_10_wksp_RD.mat', 'm012_48_wksp_RD.mat', 'm012_14_wksp_RD.mat', 'm012_33_wksp_RD.mat', 'm012_39_wksp_RD.mat', 'm012_24_wksp_RD.mat'], 'Person_006': ['m006_19_wksp_RD.mat', 'm006_18_wksp_RD.mat', 'm006_2_wksp_RD.mat', 'm006_36_wksp_RD.mat', 'm006_32_wksp_RD.mat', 'm006_13_wksp_RD.mat', 'm006_9_wksp_RD.mat', 'm006_48_wksp_RD.mat', 'm006_17_wksp_RD.mat', 'm006_27_wksp_RD.mat', 'm006_42_wksp_RD.mat', 'm006_11_wksp_RD.mat', 'm006_49_wksp_RD.mat', 'm006_34_wksp_RD.mat', 'm006_1_wksp_RD.mat', 'm006_4_wksp_RD.mat', 'm006_8_wksp_RD.mat', 'm006_26_wksp_RD.mat', 'm006_35_wksp_RD.mat', 'm006_46_wksp_RD.mat', 'm006_33_wksp_RD.mat', 'm006_44_wksp_RD.mat', 'm006_16_wksp_RD.mat', 'm006_37_wksp_RD.mat', 'm006_47_wksp_RD.mat', 'm006_3_wksp_RD.mat', 'm006_31_wksp_RD.mat', 'm006_12_wksp_RD.mat', 'm006_5_wksp_RD.mat', 'm006_28_wksp_RD.mat', 'm006_38_wksp_RD.mat', 'm006_6_wksp_RD.mat', 'm006_23_wksp_RD.mat', 'm006_40_wksp_RD.mat', 'm006_10_wksp_RD.mat', 'm006_22_wksp_RD.mat', 'm006_39_wksp_RD.mat', 'm006_15_wksp_RD.mat'], 'Person_013': ['m013_32_wksp_RD.mat', 'm013_6_wksp_RD.mat', 'm013_17_wksp_RD.mat', 'm013_24_wksp_RD.mat', 'm013_46_wksp_RD.mat', 'm013_25_wksp_RD.mat', 'm013_18_wksp_RD.mat', 'm013_19_wksp_RD.mat', 'm013_11_wksp_RD.mat', 'm013_45_wksp_RD.mat', 'm013_2_wksp_RD.mat', 'm013_1_wksp_RD.mat', 'm013_31_wksp_RD.mat', 'm013_49_wksp_RD.mat', 'm013_39_wksp_RD.mat', 'm013_38_wksp_RD.mat', 'm013_48_wksp_RD.mat', 'm013_30_wksp_RD.mat', 'm013_14_wksp_RD.mat', 'm013_3_wksp_RD.mat', 'm013_44_wksp_RD.mat', 'm013_13_wksp_RD.mat', 'm013_27_wksp_RD.mat', 'm013_36_wksp_RD.mat', 'm013_42_wksp_RD.mat', 'm013_4_wksp_RD.mat', 'm013_29_wksp_RD.mat', 'm013_35_wksp_RD.mat', 'm013_50_wksp_RD.mat', 'm013_28_wksp_RD.mat', 'm013_26_wksp_RD.mat', 'm013_37_wksp_RD.mat', 'm013_9_wksp_RD.mat', 'm013_16_wksp_RD.mat', 'm013_7_wksp_RD.mat'], 'Person_014': ['m014_15_wksp_RD.mat', 'm014_21_wksp_RD.mat', 'm014_27_wksp_RD.mat', 'm014_37_wksp_RD.mat', 'm014_9_wksp_RD.mat', 'm014_45_wksp_RD.mat', 'm014_10_wksp_RD.mat', 'm014_31_wksp_RD.mat', 'm014_50_wksp_RD.mat', 'm014_5_wksp_RD.mat', 'm014_19_wksp_RD.mat', 'm014_23_wksp_RD.mat', 'm014_22_wksp_RD.mat', 'm014_43_wksp_RD.mat', 'm014_48_wksp_RD.mat', 'm014_33_wksp_RD.mat', 'm014_38_wksp_RD.mat', 'm014_44_wksp_RD.mat', 'm014_46_wksp_RD.mat', 'm014_1_wksp_RD.mat', 'm014_13_wksp_RD.mat', 'm014_3_wksp_RD.mat', 'm014_36_wksp_RD.mat', 'm014_14_wksp_RD.mat', 'm014_42_wksp_RD.mat', 'm014_28_wksp_RD.mat', 'm014_16_wksp_RD.mat', 'm014_20_wksp_RD.mat', 'm014_8_wksp_RD.mat'], 'Person_010': ['m010_4_wksp_RD.mat', 'm010_2_wksp_RD.mat', 'm010_25_wksp_RD.mat', 'm010_6_wksp_RD.mat', 'm010_32_wksp_RD.mat', 'm010_11_wksp_RD.mat', 'm010_16_wksp_RD.mat', 'm010_23_wksp_RD.mat', 'm010_30_wksp_RD.mat', 'm010_20_wksp_RD.mat', 'm010_29_wksp_RD.mat', 'm010_7_wksp_RD.mat', 'm010_14_wksp_RD.mat', 'm010_24_wksp_RD.mat', 'm010_5_wksp_RD.mat', 'm010_12_wksp_RD.mat', 'm010_21_wksp_RD.mat', 'm010_15_wksp_RD.mat', 'm010_47_wksp_RD.mat', 'm010_48_wksp_RD.mat', 'm010_38_wksp_RD.mat', 'm010_28_wksp_RD.mat', 'm010_35_wksp_RD.mat', 'm010_49_wksp_RD.mat', 'm010_26_wksp_RD.mat', 'm010_45_wksp_RD.mat', 'm010_31_wksp_RD.mat', 'm010_17_wksp_RD.mat', 'm010_10_wksp_RD.mat', 'm010_3_wksp_RD.mat', 'm010_50_wksp_RD.mat', 'm010_36_wksp_RD.mat', 'm010_46_wksp_RD.mat', 'm010_27_wksp_RD.mat', 'm010_39_wksp_RD.mat'], 'Person_001': ['m001_48_wksp_RD.mat', 'm001_36_wksp_RD.mat', 'm001_20_wksp_RD.mat', 'm001_33_wksp_RD.mat', 'm001_28_wksp_RD.mat', 'm001_1_wksp_RD.mat', 'm001_5_wksp_RD.mat', 'm001_45_wksp_RD.mat', 'm001_2_wksp_RD.mat', 'm001_42_wksp_RD.mat', 'm001_25_wksp_RD.mat', 'm001_31_wksp_RD.mat', 'm001_16_wksp_RD.mat', 'm001_24_wksp_RD.mat', 'm001_40_wksp_RD.mat', 'm001_21_wksp_RD.mat', 'm001_38_wksp_RD.mat', 'm001_15_wksp_RD.mat', 'm001_47_wksp_RD.mat', 'm001_7_wksp_RD.mat', 'm001_34_wksp_RD.mat', 'm001_43_wksp_RD.mat', 'm001_12_wksp_RD.mat', 'm001_46_wksp_RD.mat', 'm001_23_wksp_RD.mat', 'm001_13_wksp_RD.mat', 'm001_26_wksp_RD.mat', 'm001_44_wksp_RD.mat', 'm001_37_wksp_RD.mat'], 'Person_003': ['m003_20_wksp_RD.mat', 'm003_10_wksp_RD.mat', 'm003_15_wksp_RD.mat', 'm003_7_wksp_RD.mat', 'm003_49_wksp_RD.mat', 'm003_22_wksp_RD.mat', 'm003_33_wksp_RD.mat', 'm003_9_wksp_RD.mat', 'm003_46_wksp_RD.mat', 'm003_32_wksp_RD.mat', 'm003_6_wksp_RD.mat', 'm003_18_wksp_RD.mat', 'm003_3_wksp_RD.mat', 'm003_40_wksp_RD.mat', 'm003_31_wksp_RD.mat', 'm003_45_wksp_RD.mat', 'm003_25_wksp_RD.mat', 'm003_4_wksp_RD.mat', 'm003_26_wksp_RD.mat', 'm003_34_wksp_RD.mat', 'm003_36_wksp_RD.mat', 'm003_28_wksp_RD.mat', 'm003_37_wksp_RD.mat', 'm003_21_wksp_RD.mat', 'm003_13_wksp_RD.mat', 'm003_1_wksp_RD.mat', 'm003_24_wksp_RD.mat', 'm003_2_wksp_RD.mat', 'm003_43_wksp_RD.mat', 'm003_23_wksp_RD.mat', 'm003_50_wksp_RD.mat', 'm003_8_wksp_RD.mat', 'm003_11_wksp_RD.mat', 'm003_44_wksp_RD.mat', 'm003_41_wksp_RD.mat', 'm003_35_wksp_RD.mat'], 'Person_011': ['m011_19_wksp_RD.mat', 'm011_1_wksp_RD.mat', 'm011_15_wksp_RD.mat', 'm011_21_wksp_RD.mat', 'm011_13_wksp_RD.mat', 'm011_10_wksp_RD.mat', 'm011_9_wksp_RD.mat', 'm011_38_wksp_RD.mat', 'm011_44_wksp_RD.mat', 'm011_4_wksp_RD.mat', 'm011_29_wksp_RD.mat', 'm011_43_wksp_RD.mat', 'm011_48_wksp_RD.mat', 'm011_47_wksp_RD.mat', 'm011_28_wksp_RD.mat', 'm011_5_wksp_RD.mat', 'm011_7_wksp_RD.mat', 'm011_30_wksp_RD.mat', 'm011_42_wksp_RD.mat', 'm011_22_wksp_RD.mat', 'm011_45_wksp_RD.mat', 'm011_40_wksp_RD.mat', 'm011_26_wksp_RD.mat', 'm011_27_wksp_RD.mat', 'm011_8_wksp_RD.mat'], 'Person_016': ['m016_41_wksp_RD.mat', 'm016_32_wksp_RD.mat', 'm016_10_wksp_RD.mat', 'm016_44_wksp_RD.mat', 'm016_11_wksp_RD.mat', 'm016_2_wksp_RD.mat', 'm016_21_wksp_RD.mat', 'm016_39_wksp_RD.mat', 'm016_5_wksp_RD.mat', 'm016_23_wksp_RD.mat', 'm016_45_wksp_RD.mat', 'm016_46_wksp_RD.mat', 'm016_3_wksp_RD.mat', 'm016_35_wksp_RD.mat', 'm016_30_wksp_RD.mat', 'm016_43_wksp_RD.mat', 'm016_16_wksp_RD.mat', 'm016_48_wksp_RD.mat', 'm016_26_wksp_RD.mat', 'm016_34_wksp_RD.mat', 'm016_28_wksp_RD.mat', 'm016_6_wksp_RD.mat', 'm016_8_wksp_RD.mat', 'm016_36_wksp_RD.mat', 'm016_31_wksp_RD.mat', 'm016_14_wksp_RD.mat', 'm016_37_wksp_RD.mat', 'm016_22_wksp_RD.mat', 'm016_29_wksp_RD.mat', 'm016_12_wksp_RD.mat', 'm016_9_wksp_RD.mat'], 'Person_020': ['m020_47_wksp_RD.mat', 'm020_40_wksp_RD.mat', 'm020_41_wksp_RD.mat', 'm020_2_wksp_RD.mat', 'm020_10_wksp_RD.mat', 'm020_13_wksp_RD.mat', 'm020_27_wksp_RD.mat', 'm020_18_wksp_RD.mat', 'm020_32_wksp_RD.mat', 'm020_45_wksp_RD.mat', 'm020_24_wksp_RD.mat', 'm020_44_wksp_RD.mat', 'm020_26_wksp_RD.mat', 'm020_3_wksp_RD.mat', 'm020_19_wksp_RD.mat', 'm020_17_wksp_RD.mat', 'm020_49_wksp_RD.mat', 'm020_23_wksp_RD.mat', 'm020_35_wksp_RD.mat', 'm020_25_wksp_RD.mat', 'm020_42_wksp_RD.mat', 'm020_46_wksp_RD.mat', 'm020_50_wksp_RD.mat', 'm020_48_wksp_RD.mat', 'm020_14_wksp_RD.mat', 'm020_20_wksp_RD.mat', 'm020_8_wksp_RD.mat', 'm020_39_wksp_RD.mat', 'm020_5_wksp_RD.mat', 'm020_16_wksp_RD.mat', 'm020_31_wksp_RD.mat', 'm020_28_wksp_RD.mat', 'm020_36_wksp_RD.mat', 'm020_15_wksp_RD.mat', 'm020_4_wksp_RD.mat', 'm020_11_wksp_RD.mat', 'm020_34_wksp_RD.mat'], 'Person_019': ['m019_37_wksp_RD.mat', 'm019_18_wksp_RD.mat', 'm019_9_wksp_RD.mat', 'm019_1_wksp_RD.mat', 'm019_27_wksp_RD.mat', 'm019_2_wksp_RD.mat', 'm019_21_wksp_RD.mat', 'm019_31_wksp_RD.mat', 'm019_48_wksp_RD.mat', 'm019_50_wksp_RD.mat', 'm019_20_wksp_RD.mat', 'm019_10_wksp_RD.mat', 'm019_11_wksp_RD.mat', 'm019_38_wksp_RD.mat', 'm019_15_wksp_RD.mat', 'm019_32_wksp_RD.mat', 'm019_6_wksp_RD.mat', 'm019_14_wksp_RD.mat', 'm019_5_wksp_RD.mat', 'm019_24_wksp_RD.mat', 'm019_49_wksp_RD.mat', 'm019_4_wksp_RD.mat', 'm019_7_wksp_RD.mat', 'm019_8_wksp_RD.mat', 'm019_16_wksp_RD.mat', 'm019_33_wksp_RD.mat', 'm019_23_wksp_RD.mat', 'm019_36_wksp_RD.mat', 'm019_26_wksp_RD.mat', 'm019_29_wksp_RD.mat', 'm019_46_wksp_RD.mat', 'm019_43_wksp_RD.mat', 'm019_28_wksp_RD.mat', 'm019_47_wksp_RD.mat'], 'Person_018': ['m018_13_wksp_RD.mat', 'm018_29_wksp_RD.mat', 'm018_12_wksp_RD.mat', 'm018_22_wksp_RD.mat', 'm018_41_wksp_RD.mat', 'm018_32_wksp_RD.mat', 'm018_30_wksp_RD.mat', 'm018_27_wksp_RD.mat', 'm018_15_wksp_RD.mat', 'm018_42_wksp_RD.mat', 'm018_33_wksp_RD.mat', 'm018_35_wksp_RD.mat', 'm018_38_wksp_RD.mat', 'm018_36_wksp_RD.mat', 'm018_11_wksp_RD.mat', 'm018_31_wksp_RD.mat', 'm018_4_wksp_RD.mat', 'm018_40_wksp_RD.mat', 'm018_18_wksp_RD.mat', 'm018_24_wksp_RD.mat', 'm018_2_wksp_RD.mat', 'm018_19_wksp_RD.mat', 'm018_16_wksp_RD.mat', 'm018_5_wksp_RD.mat', 'm018_48_wksp_RD.mat', 'm018_47_wksp_RD.mat', 'm018_6_wksp_RD.mat', 'm018_50_wksp_RD.mat', 'm018_46_wksp_RD.mat', 'm018_9_wksp_RD.mat', 'm018_45_wksp_RD.mat', 'm018_43_wksp_RD.mat'], 'Person_002': ['m002_34_wksp_RD.mat', 'm002_37_wksp_RD.mat', 'm002_5_wksp_RD.mat', 'm002_19_wksp_RD.mat', 'm002_46_wksp_RD.mat', 'm002_29_wksp_RD.mat', 'm002_44_wksp_RD.mat', 'm002_17_wksp_RD.mat', 'm002_32_wksp_RD.mat', 'm002_9_wksp_RD.mat', 'm002_28_wksp_RD.mat', 'm002_11_wksp_RD.mat', 'm002_42_wksp_RD.mat', 'm002_45_wksp_RD.mat', 'm002_41_wksp_RD.mat', 'm002_31_wksp_RD.mat', 'm002_14_wksp_RD.mat', 'm002_26_wksp_RD.mat', 'm002_48_wksp_RD.mat', 'm002_21_wksp_RD.mat', 'm002_43_wksp_RD.mat', 'm002_7_wksp_RD.mat', 'm002_24_wksp_RD.mat', 'm002_25_wksp_RD.mat', 'm002_12_wksp_RD.mat', 'm002_2_wksp_RD.mat', 'm002_20_wksp_RD.mat', 'm002_13_wksp_RD.mat', 'm002_23_wksp_RD.mat', 'm002_30_wksp_RD.mat', 'm002_47_wksp_RD.mat', 'm002_38_wksp_RD.mat'], 'Person_017': ['m017_35_wksp_RD.mat', 'm017_1_wksp_RD.mat', 'm017_42_wksp_RD.mat', 'm017_7_wksp_RD.mat', 'm017_45_wksp_RD.mat', 'm017_4_wksp_RD.mat', 'm017_9_wksp_RD.mat', 'm017_19_wksp_RD.mat', 'm017_13_wksp_RD.mat', 'm017_23_wksp_RD.mat', 'm017_43_wksp_RD.mat', 'm017_37_wksp_RD.mat', 'm017_46_wksp_RD.mat', 'm017_32_wksp_RD.mat', 'm017_20_wksp_RD.mat', 'm017_12_wksp_RD.mat', 'm017_48_wksp_RD.mat', 'm017_5_wksp_RD.mat', 'm017_33_wksp_RD.mat', 'm017_10_wksp_RD.mat', 'm017_16_wksp_RD.mat', 'm017_34_wksp_RD.mat', 'm017_40_wksp_RD.mat', 'm017_50_wksp_RD.mat', 'm017_11_wksp_RD.mat', 'm017_47_wksp_RD.mat', 'm017_41_wksp_RD.mat'], 'Person_004': ['m004_46_wksp_RD.mat', 'm004_50_wksp_RD.mat', 'm004_32_wksp_RD.mat', 'm004_2_wksp_RD.mat', 'm004_36_wksp_RD.mat', 'm004_17_wksp_RD.mat', 'm004_5_wksp_RD.mat', 'm004_44_wksp_RD.mat', 'm004_28_wksp_RD.mat', 'm004_19_wksp_RD.mat', 'm004_18_wksp_RD.mat', 'm004_37_wksp_RD.mat', 'm004_12_wksp_RD.mat', 'm004_4_wksp_RD.mat', 'm004_10_wksp_RD.mat', 'm004_38_wksp_RD.mat', 'm004_20_wksp_RD.mat', 'm004_35_wksp_RD.mat', 'm004_40_wksp_RD.mat', 'm004_16_wksp_RD.mat', 'm004_9_wksp_RD.mat', 'm004_3_wksp_RD.mat', 'm004_24_wksp_RD.mat', 'm004_49_wksp_RD.mat', 'm004_23_wksp_RD.mat', 'm004_22_wksp_RD.mat', 'm004_13_wksp_RD.mat'], 'Person_005': ['m005_49_wksp_RD.mat', 'm005_17_wksp_RD.mat', 'm005_26_wksp_RD.mat', 'm005_7_wksp_RD.mat', 'm005_44_wksp_RD.mat', 'm005_13_wksp_RD.mat', 'm005_45_wksp_RD.mat', 'm005_29_wksp_RD.mat', 'm005_42_wksp_RD.mat', 'm005_43_wksp_RD.mat', 'm005_16_wksp_RD.mat', 'm005_5_wksp_RD.mat', 'm005_30_wksp_RD.mat', 'm005_48_wksp_RD.mat', 'm005_37_wksp_RD.mat', 'm005_3_wksp_RD.mat', 'm005_25_wksp_RD.mat', 'm005_40_wksp_RD.mat', 'm005_20_wksp_RD.mat', 'm005_39_wksp_RD.mat', 'm005_41_wksp_RD.mat', 'm005_4_wksp_RD.mat', 'm005_34_wksp_RD.mat', 'm005_9_wksp_RD.mat', 'm005_6_wksp_RD.mat', 'm005_10_wksp_RD.mat', 'm005_33_wksp_RD.mat', 'm005_24_wksp_RD.mat', 'm005_36_wksp_RD.mat']}

test_dict = {'Person_015': ['m015_47_wksp_RD.mat', 'm015_18_wksp_RD.mat', 'm015_42_wksp_RD.mat', 'm015_1_wksp_RD.mat', 'm015_41_wksp_RD.mat', 'm015_9_wksp_RD.mat', 'm015_5_wksp_RD.mat', 'm015_11_wksp_RD.mat', 'm015_43_wksp_RD.mat', 'm015_29_wksp_RD.mat', 'm015_33_wksp_RD.mat', 'm015_12_wksp_RD.mat', 'm015_22_wksp_RD.mat', 'm015_31_wksp_RD.mat', 'm015_26_wksp_RD.mat'], 'Person_009': ['m009_5_wksp_RD.mat', 'm009_43_wksp_RD.mat', 'm009_19_wksp_RD.mat', 'm009_39_wksp_RD.mat', 'm009_13_wksp_RD.mat', 'm009_7_wksp_RD.mat', 'm009_1_wksp_RD.mat', 'm009_9_wksp_RD.mat', 'm009_20_wksp_RD.mat', 'm009_16_wksp_RD.mat', 'm009_17_wksp_RD.mat', 'm009_21_wksp_RD.mat', 'm009_46_wksp_RD.mat', 'm009_6_wksp_RD.mat', 'm009_11_wksp_RD.mat', 'm009_38_wksp_RD.mat', 'm009_12_wksp_RD.mat', 'm009_30_wksp_RD.mat', 'm009_42_wksp_RD.mat', 'm009_22_wksp_RD.mat', 'm009_48_wksp_RD.mat', 'm009_41_wksp_RD.mat', 'm009_49_wksp_RD.mat'], 'Person_007': ['m007_11_wksp_RD.mat', 'm007_13_wksp_RD.mat', 'm007_18_wksp_RD.mat', 'm007_36_wksp_RD.mat', 'm007_19_wksp_RD.mat', 'm007_15_wksp_RD.mat', 'm007_46_wksp_RD.mat', 'm007_34_wksp_RD.mat', 'm007_16_wksp_RD.mat', 'm007_21_wksp_RD.mat', 'm007_41_wksp_RD.mat', 'm007_25_wksp_RD.mat', 'm007_17_wksp_RD.mat', 'm007_29_wksp_RD.mat', 'm007_26_wksp_RD.mat', 'm007_12_wksp_RD.mat', 'm007_39_wksp_RD.mat'], 'Person_008': ['m008_6_wksp_RD.mat', 'm008_26_wksp_RD.mat', 'm008_19_wksp_RD.mat', 'm008_45_wksp_RD.mat', 'm008_43_wksp_RD.mat', 'm008_50_wksp_RD.mat', 'm008_40_wksp_RD.mat', 'm008_35_wksp_RD.mat', 'm008_21_wksp_RD.mat', 'm008_27_wksp_RD.mat', 'm008_37_wksp_RD.mat', 'm008_36_wksp_RD.mat', 'm008_29_wksp_RD.mat', 'm008_8_wksp_RD.mat', 'm008_39_wksp_RD.mat', 'm008_41_wksp_RD.mat', 'm008_11_wksp_RD.mat'], 'Person_012': ['m012_38_wksp_RD.mat', 'm012_29_wksp_RD.mat', 'm012_27_wksp_RD.mat', 'm012_35_wksp_RD.mat', 'm012_25_wksp_RD.mat', 'm012_2_wksp_RD.mat', 'm012_8_wksp_RD.mat', 'm012_45_wksp_RD.mat', 'm012_46_wksp_RD.mat', 'm012_12_wksp_RD.mat', 'm012_1_wksp_RD.mat', 'm012_16_wksp_RD.mat', 'm012_22_wksp_RD.mat'], 'Person_006': ['m006_41_wksp_RD.mat', 'm006_21_wksp_RD.mat', 'm006_14_wksp_RD.mat', 'm006_24_wksp_RD.mat', 'm006_30_wksp_RD.mat', 'm006_7_wksp_RD.mat', 'm006_20_wksp_RD.mat', 'm006_45_wksp_RD.mat', 'm006_43_wksp_RD.mat', 'm006_29_wksp_RD.mat', 'm006_50_wksp_RD.mat', 'm006_25_wksp_RD.mat'], 'Person_013': ['m013_20_wksp_RD.mat', 'm013_23_wksp_RD.mat', 'm013_8_wksp_RD.mat', 'm013_41_wksp_RD.mat', 'm013_21_wksp_RD.mat', 'm013_15_wksp_RD.mat', 'm013_10_wksp_RD.mat', 'm013_33_wksp_RD.mat', 'm013_22_wksp_RD.mat', 'm013_43_wksp_RD.mat', 'm013_5_wksp_RD.mat', 'm013_47_wksp_RD.mat', 'm013_34_wksp_RD.mat', 'm013_40_wksp_RD.mat', 'm013_12_wksp_RD.mat'], 'Person_014': ['m014_25_wksp_RD.mat', 'm014_2_wksp_RD.mat', 'm014_26_wksp_RD.mat', 'm014_17_wksp_RD.mat', 'm014_49_wksp_RD.mat', 'm014_11_wksp_RD.mat', 'm014_7_wksp_RD.mat', 'm014_4_wksp_RD.mat', 'm014_40_wksp_RD.mat', 'm014_39_wksp_RD.mat', 'm014_12_wksp_RD.mat', 'm014_24_wksp_RD.mat', 'm014_32_wksp_RD.mat', 'm014_35_wksp_RD.mat', 'm014_18_wksp_RD.mat', 'm014_6_wksp_RD.mat', 'm014_29_wksp_RD.mat', 'm014_30_wksp_RD.mat', 'm014_47_wksp_RD.mat', 'm014_41_wksp_RD.mat', 'm014_34_wksp_RD.mat'], 'Person_010': ['m010_8_wksp_RD.mat', 'm010_42_wksp_RD.mat', 'm010_18_wksp_RD.mat', 'm010_19_wksp_RD.mat', 'm010_9_wksp_RD.mat', 'm010_1_wksp_RD.mat', 'm010_13_wksp_RD.mat', 'm010_43_wksp_RD.mat', 'm010_44_wksp_RD.mat', 'm010_33_wksp_RD.mat', 'm010_37_wksp_RD.mat', 'm010_34_wksp_RD.mat', 'm010_22_wksp_RD.mat', 'm010_40_wksp_RD.mat', 'm010_41_wksp_RD.mat'], 'Person_001': ['m001_30_wksp_RD.mat', 'm001_14_wksp_RD.mat', 'm001_19_wksp_RD.mat', 'm001_18_wksp_RD.mat', 'm001_32_wksp_RD.mat', 'm001_8_wksp_RD.mat', 'm001_41_wksp_RD.mat', 'm001_11_wksp_RD.mat', 'm001_27_wksp_RD.mat', 'm001_6_wksp_RD.mat', 'm001_29_wksp_RD.mat', 'm001_39_wksp_RD.mat', 'm001_3_wksp_RD.mat', 'm001_9_wksp_RD.mat', 'm001_22_wksp_RD.mat', 'm001_4_wksp_RD.mat', 'm001_35_wksp_RD.mat', 'm001_17_wksp_RD.mat', 'm001_10_wksp_RD.mat'], 'Person_003': ['m003_30_wksp_RD.mat', 'm003_19_wksp_RD.mat', 'm003_14_wksp_RD.mat', 'm003_39_wksp_RD.mat', 'm003_48_wksp_RD.mat', 'm003_12_wksp_RD.mat', 'm003_16_wksp_RD.mat', 'm003_17_wksp_RD.mat', 'm003_29_wksp_RD.mat', 'm003_38_wksp_RD.mat', 'm003_47_wksp_RD.mat', 'm003_27_wksp_RD.mat', 'm003_5_wksp_RD.mat', 'm003_42_wksp_RD.mat'], 'Person_011': ['m011_14_wksp_RD.mat', 'm011_17_wksp_RD.mat', 'm011_12_wksp_RD.mat', 'm011_16_wksp_RD.mat', 'm011_20_wksp_RD.mat', 'm011_18_wksp_RD.mat', 'm011_11_wksp_RD.mat', 'm011_2_wksp_RD.mat', 'm011_36_wksp_RD.mat', 'm011_25_wksp_RD.mat', 'm011_33_wksp_RD.mat', 'm011_34_wksp_RD.mat', 'm011_24_wksp_RD.mat', 'm011_3_wksp_RD.mat', 'm011_31_wksp_RD.mat', 'm011_6_wksp_RD.mat', 'm011_39_wksp_RD.mat', 'm011_50_wksp_RD.mat', 'm011_23_wksp_RD.mat', 'm011_49_wksp_RD.mat', 'm011_32_wksp_RD.mat', 'm011_37_wksp_RD.mat', 'm011_46_wksp_RD.mat', 'm011_41_wksp_RD.mat', 'm011_35_wksp_RD.mat'], 'Person_016': ['m016_24_wksp_RD.mat', 'm016_49_wksp_RD.mat', 'm016_25_wksp_RD.mat', 'm016_33_wksp_RD.mat', 'm016_7_wksp_RD.mat', 'm016_15_wksp_RD.mat', 'm016_17_wksp_RD.mat', 'm016_27_wksp_RD.mat', 'm016_1_wksp_RD.mat', 'm016_40_wksp_RD.mat', 'm016_4_wksp_RD.mat', 'm016_47_wksp_RD.mat', 'm016_18_wksp_RD.mat', 'm016_19_wksp_RD.mat', 'm016_13_wksp_RD.mat', 'm016_38_wksp_RD.mat', 'm016_50_wksp_RD.mat', 'm016_42_wksp_RD.mat', 'm016_20_wksp_RD.mat'], 'Person_020': ['m020_6_wksp_RD.mat', 'm020_9_wksp_RD.mat', 'm020_12_wksp_RD.mat', 'm020_43_wksp_RD.mat', 'm020_30_wksp_RD.mat', 'm020_33_wksp_RD.mat', 'm020_21_wksp_RD.mat', 'm020_7_wksp_RD.mat', 'm020_29_wksp_RD.mat', 'm020_38_wksp_RD.mat', 'm020_22_wksp_RD.mat', 'm020_1_wksp_RD.mat', 'm020_37_wksp_RD.mat'], 'Person_019': ['m019_30_wksp_RD.mat', 'm019_17_wksp_RD.mat', 'm019_39_wksp_RD.mat', 'm019_13_wksp_RD.mat', 'm019_35_wksp_RD.mat', 'm019_19_wksp_RD.mat', 'm019_12_wksp_RD.mat', 'm019_34_wksp_RD.mat', 'm019_41_wksp_RD.mat', 'm019_45_wksp_RD.mat', 'm019_3_wksp_RD.mat', 'm019_25_wksp_RD.mat', 'm019_22_wksp_RD.mat', 'm019_44_wksp_RD.mat', 'm019_40_wksp_RD.mat', 'm019_42_wksp_RD.mat'], 'Person_018': ['m018_21_wksp_RD.mat', 'm018_28_wksp_RD.mat', 'm018_34_wksp_RD.mat', 'm018_20_wksp_RD.mat', 'm018_39_wksp_RD.mat', 'm018_23_wksp_RD.mat', 'm018_26_wksp_RD.mat', 'm018_25_wksp_RD.mat', 'm018_37_wksp_RD.mat', 'm018_10_wksp_RD.mat', 'm018_3_wksp_RD.mat', 'm018_17_wksp_RD.mat', 'm018_14_wksp_RD.mat', 'm018_1_wksp_RD.mat', 'm018_44_wksp_RD.mat', 'm018_7_wksp_RD.mat', 'm018_49_wksp_RD.mat', 'm018_8_wksp_RD.mat'], 'Person_002': ['m002_1_wksp_RD.mat', 'm002_35_wksp_RD.mat', 'm002_39_wksp_RD.mat', 'm002_8_wksp_RD.mat', 'm002_15_wksp_RD.mat', 'm002_3_wksp_RD.mat', 'm002_10_wksp_RD.mat', 'm002_40_wksp_RD.mat', 'm002_36_wksp_RD.mat', 'm002_18_wksp_RD.mat', 'm002_4_wksp_RD.mat', 'm002_49_wksp_RD.mat', 'm002_33_wksp_RD.mat', 'm002_16_wksp_RD.mat', 'm002_22_wksp_RD.mat', 'm002_50_wksp_RD.mat', 'm002_27_wksp_RD.mat', 'm002_6_wksp_RD.mat'], 'Person_017': ['m017_6_wksp_RD.mat', 'm017_8_wksp_RD.mat', 'm017_21_wksp_RD.mat', 'm017_15_wksp_RD.mat', 'm017_18_wksp_RD.mat', 'm017_22_wksp_RD.mat', 'm017_17_wksp_RD.mat', 'm017_30_wksp_RD.mat', 'm017_29_wksp_RD.mat', 'm017_25_wksp_RD.mat', 'm017_28_wksp_RD.mat', 'm017_31_wksp_RD.mat', 'm017_14_wksp_RD.mat', 'm017_24_wksp_RD.mat', 'm017_44_wksp_RD.mat', 'm017_36_wksp_RD.mat', 'm017_2_wksp_RD.mat', 'm017_39_wksp_RD.mat', 'm017_49_wksp_RD.mat', 'm017_3_wksp_RD.mat', 'm017_26_wksp_RD.mat', 'm017_27_wksp_RD.mat', 'm017_38_wksp_RD.mat'], 'Person_004': ['m004_27_wksp_RD.mat', 'm004_31_wksp_RD.mat', 'm004_15_wksp_RD.mat', 'm004_11_wksp_RD.mat', 'm004_45_wksp_RD.mat', 'm004_8_wksp_RD.mat', 'm004_48_wksp_RD.mat', 'm004_14_wksp_RD.mat', 'm004_29_wksp_RD.mat', 'm004_41_wksp_RD.mat', 'm004_7_wksp_RD.mat', 'm004_33_wksp_RD.mat', 'm004_47_wksp_RD.mat', 'm004_1_wksp_RD.mat', 'm004_30_wksp_RD.mat', 'm004_21_wksp_RD.mat', 'm004_26_wksp_RD.mat', 'm004_42_wksp_RD.mat', 'm004_25_wksp_RD.mat', 'm004_6_wksp_RD.mat', 'm004_43_wksp_RD.mat', 'm004_34_wksp_RD.mat', 'm004_39_wksp_RD.mat'], 'Person_005': ['m005_21_wksp_RD.mat', 'm005_19_wksp_RD.mat', 'm005_2_wksp_RD.mat', 'm005_11_wksp_RD.mat', 'm005_18_wksp_RD.mat', 'm005_1_wksp_RD.mat', 'm005_46_wksp_RD.mat', 'm005_15_wksp_RD.mat', 'm005_31_wksp_RD.mat', 'm005_47_wksp_RD.mat', 'm005_38_wksp_RD.mat', 'm005_28_wksp_RD.mat', 'm005_14_wksp_RD.mat', 'm005_32_wksp_RD.mat', 'm005_8_wksp_RD.mat', 'm005_50_wksp_RD.mat', 'm005_35_wksp_RD.mat', 'm005_22_wksp_RD.mat', 'm005_12_wksp_RD.mat', 'm005_23_wksp_RD.mat', 'm005_27_wksp_RD.mat']}


################## Taking Entire Signal ###################
X_train,Y_train, X_test, Y_test = epoch_data(data,window = None, train_dict=train_dict,test_dict=test_dict)
print("Array Shapes")
print(np.array(X_train).shape)
print(np.array(X_test).shape)
X_train_padded = []
for x in X_train:
  for xx in x:
    if xx.shape[0] < 12000:
      print(xx.shape)
      padding = 12000 - xx.shape[0]
      left_padding = padding //2
      right_padding = padding - left_padding
      padded_array = np.pad(xx, ((left_padding, right_padding), (0,0)), mode='constant')
      print(padded_array.shape)
      X_train_padded.append(padded_array)
    else:
      X_train_padded.append(xx)

X_test_padded = []
for x in X_test:
  for xx in x:
    if xx.shape[0] < 12000:
      print(xx.shape)
      padding = 12000 - xx.shape[0]
      left_padding = padding //2
      right_padding = padding - left_padding
      padded_array = np.pad(xx, ((left_padding, right_padding), (0,0)), mode='constant')
      print(padded_array.shape)
      X_test_padded.append(padded_array)
    else:
      X_test_padded.append(xx)

X_test = np.array(X_test_padded)
X_train = np.array(X_train_padded)
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(-1,1)
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(-1,1)

print(X_train.shape)

"""
plt.title('ECG Recordings')
plt.xlabel('Time')
plt.plot(X_train[110,:100,:], label = Y_train[110])
plt.plot(X_train[250,:100,:], label = Y_train[250])
plt.plot(X_train[310,:100,:], label = Y_train[310])
plt.legend()
plt.show()
"""
np.save('/Users/sultm0a/Documents/Research/Cyclo_Stationary_ML/X_train_Raw_ECG.npy', X_train)
np.save('/Users/sultm0a/Documents/Research/Cyclo_Stationary_ML/Y_train_Raw_ECG.npy',Y_train)
np.save('/Users/sultm0a/Documents/Research/Cyclo_Stationary_ML/X_test_Raw_ECG.npy', X_test)
np.save('/Users/sultm0a/Documents/Research/Cyclo_Stationary_ML/Y_test_Raw_ECG.npy',Y_test)
