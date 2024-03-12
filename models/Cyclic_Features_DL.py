from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU,ReLU
from tensorflow.keras.layers import BatchNormalization,Dropout
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from scipy.interpolate import interp1d
sns.set_context('paper')
from sklearn.model_selection import GridSearchCV
np.random.seed(150)
############ Reading Data #############
#######################################
data_path = '/Users/sultm0a/Downloads/RYXA_DATA_ALL_Particpants_During_Music.csv'
ryxa_data = pd.read_csv(data_path)
number_of_Participants = len(ryxa_data['Participant'].unique())
print("Number of Participants = {}".format(number_of_Participants))
RYXA_DATA = ryxa_data.iloc[:,2:]
print(RYXA_DATA.head())

############# Converting Complex Numbers to Magnitude ###########
print("Before Conversion to Magnitude")
print(50*"#")
RYXA_DATA['RYXA'] = np.abs(RYXA_DATA.iloc[:,0].astype(complex))
print("After Conversion to Magnitude")
print(50*"#")
print(RYXA_DATA.head())

################## EDA & Data Visualization #####################
#################################################################

#sns.displot(data=RYXA_DATA, x='RYXA', hue='Participant', kind='kde', fill=True, aspect=1.5).set(title = "Distribution of RYXA across Participants")
#plt.show()

################# Extracting Data of each participant ###########
#################################################################

# For every_participant Creating Training and Test Data
Train = []
Test = []
training_dict = {}
test_dict     = {}
for part in RYXA_DATA['Participant'].unique():
  data = RYXA_DATA[RYXA_DATA['Participant'] == part]
  training_dict[part] = []
  test_dict[part]     = []
  for recording in data['Record'].unique():
    rand_n = np.random.rand()
    if rand_n > 0.20:
      Train.append(data[data['Record'] == recording])
      training_dict[part].append(recording)
    else:
      Test.append(data[data['Record'] == recording])
      test_dict[part].append(recording)

Training_data = pd.concat(Train)
Test_data = pd.concat(Test)
print('Training Dict')
print(training_dict)
print('Test Dict')
print(test_dict)

max_l = []
min_l = []
length = []
for part in Training_data['Participant'].unique():
  data = Training_data[Training_data['Participant'] == part]
  part_max = []
  part_min = []
  for record in data['Record'].unique():
    rec_data = data[data['Record'] == record]
    maxim = max(rec_data['TAU'])
    minim = min(rec_data['TAU'])
    part_max.append(maxim)
    part_min.append(minim)
    length.append(rec_data.shape[0])
  max_l.append(part_max)
  min_l.append(part_min)

X_train_RYXA = Training_data[['RYXA']]
Y_train_RYXA = Training_data['Participant']

X_test_RYXA = Test_data[['RYXA']]
Y_test_RYXA = Test_data['Participant']

###################### Scaling the data ################
########################################################

sc = StandardScaler()
sc = sc.fit(X_train_RYXA)
X_train_RYXA_scaled = sc.transform(X_train_RYXA)
X_test_RYXA_scaled = sc.transform(X_test_RYXA)
X_train_RYXA_scaled = X_train_RYXA_scaled.reshape(-1,256)
X_test_RYXA_scaled = X_test_RYXA_scaled.reshape(-1,256)
Y_train_RYXA = np.array(Y_train_RYXA).reshape(-1,256)
Y_test_RYXA = np.array(Y_test_RYXA).reshape(-1,256)

print(" Number of Training Samples X {}".format(X_train_RYXA_scaled.shape))
print(" Number of Training Samples Y {}".format(Y_train_RYXA.shape[0]))
print(" Number of Test     Samples X {}".format(X_test_RYXA_scaled.shape))
print(" Number of Test     Samples X {}".format(Y_test_RYXA.shape[0]))

######################### Shuffling the Data  ############
##########################################################
X_train_RYXA_scaled_shuffled = np.zeros_like(X_train_RYXA_scaled)
X_test_RYXA_scaled_shuffled = np.zeros_like(X_test_RYXA_scaled)
Y_train_shuffled = np.zeros_like(Y_train_RYXA)
Y_test_shuffled = np.zeros_like(Y_test_RYXA)

training_list_indexes = np.arange(X_train_RYXA_scaled.shape[0])
test_list_indexes = np.arange(X_test_RYXA_scaled.shape[0])
random.shuffle(training_list_indexes)
random.shuffle(test_list_indexes)
shuffled_training_ind = training_list_indexes
shuffled_test_ind = test_list_indexes

for i,j in zip(shuffled_training_ind,np.arange(len(shuffled_training_ind))):

  X_train_RYXA_scaled_shuffled[j,:] = X_train_RYXA_scaled[i,:]
  Y_train_shuffled[j,:] = Y_train_RYXA[i,:]
for i,j in zip(shuffled_test_ind,np.arange(len(shuffled_test_ind))):
  X_test_RYXA_scaled_shuffled[j,:] = X_test_RYXA_scaled[i,:]
  Y_test_shuffled[j,:] = Y_test_RYXA[i,:]

X_train_RYXA_scaled  = X_train_RYXA_scaled_shuffled
X_test_RYXA_scaled  = X_test_RYXA_scaled_shuffled
Y_train_RYXA = Y_train_shuffled
Y_test_RYXA = Y_test_shuffled

Y_train_RYXA_enc = Y_train_RYXA[:,0]
Y_train_RYXA_enc = Y_train_RYXA_enc.reshape(-1)

Y_test_RYXA_enc = Y_test_RYXA[:,0]
Y_test_RYXA_enc = Y_test_RYXA_enc.reshape(-1)

################## Label Encoding the Categories ###########

lb = LabelEncoder()
lb = lb.fit(Y_train_RYXA_enc)
Y_train_RYXA_enc = lb.transform(Y_train_RYXA_enc)
Y_test_RYXA_enc = lb.transform(Y_test_RYXA_enc)
print("The Classes \n{}".format(lb.classes_))
lb_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
print("Mapping of Lables \n {}".format(lb_name_mapping))
print(50*"#")


n_inputs = 256
visible = Input(shape=(n_inputs,),name  = 'Input' )
# encoder level 1
e = Dense(128,kernel_regularizer=regularizers.L2(1e-3),
    bias_regularizer=regularizers.L2(1e-3),
    activity_regularizer=regularizers.L2(1e-3), name  = 'Dense1')(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2

e = Dense(64, kernel_regularizer=regularizers.L2(1e-3),
    bias_regularizer=regularizers.L2(1e-3),
    activity_regularizer=regularizers.L2(1e-3),name  = 'Dense2')(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)

# encoder level 1
e = Dense(32,kernel_regularizer=regularizers.L2(1e-3),
    bias_regularizer=regularizers.L2(1e-3),
    activity_regularizer=regularizers.L2(1e-3),name  = 'Dense3')(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = 2
bottleneck = Dense(n_bottleneck,activation = 'leaky_relu',kernel_regularizer=regularizers.L2(1e-3),
    bias_regularizer=regularizers.L2(1e-2),
    activity_regularizer=regularizers.L2(1e-2),name  = 'bottle_neck')(e)
bottleneck = BatchNormalization()(bottleneck)
# output layer
output = Dense(20, activation='softmax')(bottleneck)
# define autoencoder model
model = Model(inputs=visible, outputs=output)

# compile autoencoder model

checkpoint = ModelCheckpoint('/Users/sultm0a/Documents/Research/Cyclo_Stationary_ML/models/best_model_cyclic.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience= 50, mode='max', verbose=1)

optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# plot the autoencoder
#plot_model(model, 'Classifier.png', show_shapes=True)
history = model.fit(
    X_train_RYXA_scaled, Y_train_RYXA_enc,
    epochs=400, batch_size=32, verbose=2,
    validation_data=(X_test_RYXA_scaled, Y_test_RYXA_enc),
    callbacks=[checkpoint, early_stopping]
)
best_model = load_model('/Users/sultm0a/Documents/Research/Cyclo_Stationary_ML/models/best_model_cyclic.h5')
predictions= best_model.predict(X_test_RYXA_scaled)
predictions = np.argmax(predictions, axis= 1)
print(classification_report(Y_test_RYXA_enc,predictions ))

bottleneck_model = Model(inputs=best_model.input, outputs=best_model.get_layer('bottle_neck').output)
X = np.vstack((X_train_RYXA_scaled,X_test_RYXA_scaled))
Y = np.vstack((Y_train_RYXA_enc.reshape(-1,1),Y_test_RYXA_enc.reshape(-1,1)))
print(X.shape)
print(Y.shape)
bottleneck_output = bottleneck_model.predict(X)
df  = pd.DataFrame(bottleneck_output)
df['Labels'] = Y
custom_palette = sns.color_palette('husl', n_colors=20)
print(df.head())
sns.pairplot(df,hue ='Labels',palette=custom_palette )
plt.show()