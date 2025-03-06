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

data_path = 'data_path'
fourier_coeffs = pd.read_csv(data_path)
print(fourier_coeffs.head())
fourier_coeffs['Fourier_Coeffs'] = np.abs(fourier_coeffs['Fourier_Coeffs'].astype(complex))
fourier_coeffs['Harmonics'] = np.round(fourier_coeffs['Harmonics'], 3)
number_of_Participants = len(fourier_coeffs['Participant'].unique())
print("Number of Participants = {}".format(number_of_Participants))
FC_DATA = fourier_coeffs.iloc[:,2:]
#FC_DATA = FC_DATA[FC_DATA['Record'] != 'm018_38_wksp_RD.mat']
print(FC_DATA.head())

################# Extracting Data of each participant ###########
#################################################################

def pad_df(df):
    common_freq_range = np.arange(-0.4, 0.4, step=0.00313)
    # Step 2: Interpolate Fourier coefficients to match common frequencies
    harmonics = df['Harmonics'].values
    coeffs = df['Fourier_Coeffs'].values

    interpolator = interp1d(harmonics, coeffs, kind='linear', fill_value="extrapolate", bounds_error=False)
    harmonized_coeffs = interpolator(common_freq_range)
    zeros = np.zeros(shape = (len(common_freq_range),4))
    data_frame = pd.DataFrame(zeros, columns = df.columns)
    data_frame['Fourier_Coeffs'] = harmonized_coeffs
    data_frame['Harmonics']      = common_freq_range
    data_frame['Record']         = df.iloc[0,2]
    data_frame['Participant']    = df.iloc[0,3]
    #fig,ax = plt.subplots(nrows=1,ncols=2)
    #ax[0].plot(df['Harmonics'], df['Fourier_Coeffs'], lw = 2, c= 'r')
    #ax[1].plot(data_frame['Harmonics'], data_frame['Fourier_Coeffs'], lw = 2, c= 'b')
    #plt.show()
    return data_frame
        # Create a new DataFrame with harmonized coefficients




# For every_participant Creating Training and Test Data
Train = []
Test = []
training_dict = {}
test_dict     = {}
for part in FC_DATA['Participant'].unique():
  data = FC_DATA[FC_DATA['Participant'] == part]
  training_dict[part] = []
  test_dict[part]     = []
  for recording in data['Record'].unique():
    rand_n = np.random.rand()
    if rand_n > 0.20:
      df = data[data['Record'] == recording]
      df = pad_df(df)
      print(df.shape)
      Train.append(df)
      training_dict[part].append(recording)
    else:
      df = data[data['Record'] == recording]
      df = pad_df(df)
      print(df.shape)
      Test.append(df)
      test_dict[part].append(recording)

Training_data = pd.concat(Train)
Test_data = pd.concat(Test)




training_list = []
test_list = []
length = []
for part in Training_data['Participant'].unique():
  data = Training_data[Training_data['Participant'] == part]
  for record in data['Record'].unique():
    if record != 0:
        rec_data = data[data['Record'] == record]
        training_list.append(rec_data['Harmonics'].shape[0])
        if rec_data['Harmonics'].shape[0] >= 12000:
            print(record)
            print(part)

    

for part in Test_data['Participant'].unique():
  data = Test_data[Test_data['Participant'] == part]
  for record in data['Record'].unique():
    if record != 0:
        rec_data = data[data['Record'] == record]
        test_list.append(rec_data['Harmonics'].shape[0])
print('Training Max {}'.format(max(training_list)))
print('Training Min {}'.format(min(training_list)))
print('Test     Max {}'.format(max(test_list)))
print('Test     Min {}'.format(min(test_list)))

X_train_RYXA = Training_data['Fourier_Coeffs'].values
Y_train_RYXA = Training_data['Participant']

X_test_RYXA = Test_data['Fourier_Coeffs'].values
Y_test_RYXA = Test_data['Participant']

X_train_RYXA = X_train_RYXA.reshape(-1,1)
X_test_RYXA = X_test_RYXA.reshape(-1,1)



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

cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA classifier
lda_classifier = LinearDiscriminantAnalysis()

# Define the parameter grid to search through
param_grid_lda = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9],
    'n_components': [None, 1, 2,3]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search_lda = GridSearchCV(estimator=lda_classifier, param_grid=param_grid_lda,
                               scoring='accuracy', cv=cv_stratified, verbose=2)
grid_search_lda.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Get the best parameters from the grid search
best_params_lda = grid_search_lda.best_params_
print('LDA Best Parameters')
print(best_params_lda)

# Train the model with the best parameters on the entire training set
best_lda_classifier = LinearDiscriminantAnalysis(**best_params_lda)
best_lda_classifier.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Make predictions on the test set
y_pred_lda = best_lda_classifier.predict(X_test_RYXA_scaled)

# Compute the confusion matrix
conf_matrix_lda = classification_report(Y_test_RYXA_enc, y_pred_lda)

# Print the best hyperparameters
print("Best Hyperparameters (LDA):", best_params_lda)

# Print the confusion matrix
print("Classification Report (LDA)")
print(conf_matrix_lda)

# Compute and print accuracy
accuracy_lda = accuracy_score(Y_test_RYXA_enc, y_pred_lda)
print("Accuracy (LDA):", accuracy_lda)

######################### Random Forest Cross Validation ###################


# Define the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Define the parameter grid to search through
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a stratified k-fold cross-validator


# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, 
                           scoring='accuracy', cv=cv_stratified, verbose= 2)
grid_search.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Train the model with the best parameters on the entire training set
best_rf_classifier = RandomForestClassifier(**best_params)
best_rf_classifier.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Make predictions on the test set
y_pred = best_rf_classifier.predict(X_test_RYXA_scaled)

# Compute the confusion matrix
conf_matrix = classification_report(Y_test_RYXA_enc, y_pred)

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Print the confusion matrix
print("Classification Report")
print(conf_matrix)

# Compute and print accuracy
accuracy = accuracy_score(Y_test_RYXA_enc, y_pred)
print("Accuracy:", accuracy)

feature_importances = best_rf_classifier.feature_importances_
plt.plot(feature_importances, lw  = 2)
plt.show()

######################### KNN ###############################

knn_classifier = KNeighborsClassifier()

# Define the parameter grid to search through
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search_knn = GridSearchCV(estimator=knn_classifier, param_grid=param_grid_knn, 
                                scoring='accuracy', cv=cv_stratified, verbose=2)
grid_search_knn.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Get the best parameters from the grid search
best_params_knn = grid_search_knn.best_params_
print('KNN Best Parameters')
print(best_params_knn)

# Train the model with the best parameters on the entire training set
best_knn_classifier = KNeighborsClassifier(**best_params_knn)
best_knn_classifier.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Make predictions on the test set
y_pred_knn = best_knn_classifier.predict(X_test_RYXA_scaled)

# Compute the confusion matrix
conf_matrix_knn = classification_report(Y_test_RYXA_enc, y_pred_knn)

# Print the best hyperparameters
print("Best Hyperparameters (KNN):", best_params_knn)

# Print the confusion matrix
print("Classification Report (KNN)")
print(conf_matrix_knn)

# Compute and print accuracy
accuracy_knn = accuracy_score(Y_test_RYXA_enc, y_pred_knn)
print("Accuracy (KNN):", accuracy_knn)




########### SVM######################


# SVM classifier
svm_classifier = SVC()

# Define the parameter grid to search through
param_grid_svm = {
    'C': [0.1, 0.3, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}

# Perform GridSearchCV for hyperparameter tuning
grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid_svm, 
                               scoring='accuracy', cv=cv_stratified, verbose=2)
grid_search_svm.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Get the best parameters from the grid search
best_params_svm = grid_search_svm.best_params_
print('SVM Best Parameters')
print(best_params_svm)

# Train the model with the best parameters on the entire training set
best_svm_classifier = SVC(**best_params_svm)
best_svm_classifier.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Make predictions on the test set
y_pred_svm = best_svm_classifier.predict(X_test_RYXA_scaled)

# Compute the confusion matrix
conf_matrix_svm = classification_report(Y_test_RYXA_enc, y_pred_svm)

# Print the best hyperparameters
print("Best Hyperparameters (SVM):", best_params_svm)

# Print the confusion matrix
print("Classification Report (SVM)")
print(conf_matrix_svm)

# Compute and print accuracy
accuracy_svm = accuracy_score(Y_test_RYXA_enc, y_pred_svm)
print("Accuracy (SVM):", accuracy_svm)
################### XGBOOST #######################


# XGBoost classifier
xgb_classifier = XGBClassifier()

# Define the parameter grid to search through
param_grid_xgb = {
    'learning_rate': [0.01, 0.001, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10,20],
    'subsample': [0.1,0.3,0.5,1],
}

# Perform GridSearchCV for hyperparameter tuning
grid_search_xgb = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid_xgb,
                               scoring='accuracy', cv=cv_stratified, verbose=2)
grid_search_xgb.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Get the best parameters from the grid search
best_params_xgb = grid_search_xgb.best_params_
print('XGBoost Best Parameters')
print(best_params_xgb)

# Train the model with the best parameters on the entire training set
best_xgb_classifier = XGBClassifier(**best_params_xgb)
best_xgb_classifier.fit(X_train_RYXA_scaled, Y_train_RYXA_enc)

# Make predictions on the test set
y_pred_xgb = best_xgb_classifier.predict(X_test_RYXA_scaled)

# Compute the confusion matrix
conf_matrix_xgb = classification_report(Y_test_RYXA_enc, y_pred_xgb)

# Print the best hyperparameters
print("Best Hyperparameters (XGBoost):", best_params_xgb)

# Print the confusion matrix
print("Classification Report (XGBoost)")
print(conf_matrix_xgb)

# Compute and print accuracy
accuracy_xgb = accuracy_score(Y_test_RYXA_enc, y_pred_xgb)
print("Accuracy (XGBoost):", accuracy_xgb)



