import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")
Window_Length = 100

#Reading the data from csv
df = pd.read_csv("Acc_Gyro_F.csv")
test_input = []
test_output = []

#Solving time series problem using window method
i = 0
while(i <= df.shape[0] - Window_Length):
    temp = df[i:i+Window_Length]
    inp = temp.drop(temp[['Output']],axis = 1)
    out = temp[['Output']]
    test_input.append(inp)
    test_output.append(out.iloc[0])
    i = i + int(Window_Length/2)

#Extracting the nessecary features in a single window
mean = pd.DataFrame()
standard_deviation = pd.DataFrame()
mad = pd.DataFrame()
avgdiff = pd.DataFrame()
maximum = pd.DataFrame()
minimum = pd.DataFrame()
var = pd.DataFrame()

for windows in test_input:
    df_mean = pd.DataFrame(windows.mean()).transpose()
    df_std = pd.DataFrame(windows.std()).transpose()
    df_max = pd.DataFrame(windows.max()).transpose()
    df_med_abs = pd.DataFrame(windows.mad()).transpose()
    df_avg_diff = pd.DataFrame(windows.diff().dropna().div(10).mean()).transpose()
    df_min = pd.DataFrame(windows.min()).transpose()
    df_var = pd.DataFrame(windows.var()).transpose()
    mean = pd.concat([mean, df_mean])
    standard_deviation = pd.concat([standard_deviation, df_std])
    mad = pd.concat([mad, df_med_abs])
    maximum = pd.concat([maximum, df_max])
    minimum = pd.concat([minimum, df_min])
    avgdiff = pd.concat([avgdiff, df_avg_diff])
    var = pd.concat([var, df_var])

mean = mean.reset_index(drop=True)
standard_deviation = standard_deviation.reset_index(drop=True)
mad = mad.reset_index(drop=True)
avgdiff = avgdiff.reset_index(drop=True)
maximum = maximum.reset_index(drop=True)
minimum = minimum.reset_index(drop=True)
var = var.reset_index(drop=True)


var.columns = [x + "var" for x in var.columns]
maximum.columns = [x + "max" for x in maximum.columns]
avgdiff.columns = [x + "avgdiff" for x in avgdiff.columns]
mean.columns = [x + "mean" for x in mean.columns]
standard_deviation.columns = [x + "std" for x in standard_deviation.columns]
mad.columns = [x + "mad" for x in mad.columns]
minimum.columns = [x + "min" for x in minimum.columns]
feature_columns = mean.join([standard_deviation, maximum, minimum, var, mad])
#print(feature_columns)


feature_columns['Output'] = np.array(test_output)

#Spliting the data in 75:25
train, test = train_test_split(feature_columns, test_size=0.25, shuffle=True)
in_train = train.drop(train[['Output']],axis=1)
out_train = train[['Output']]

in_test = test.drop(test[['Output']],axis=1)
out_test = test[['Output']]

scaler = StandardScaler()
scaler.fit(in_train)
in_train = scaler.transform(in_train)
in_test = scaler.transform(in_test)

#Using KNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(in_train, out_train)

#Using Logistic Regression
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(in_train, out_train)

#Using SVM classifier
svc = SVC(gamma = 'auto')
svc.fit(in_train, out_train)

SVM = []
KNN = []
LR = []
for i in range(10):
    SVM.append(svc.score(in_test, out_test))
    KNN.append(clf.score(in_test, out_test))
    LR.append(neigh.score(in_test, out_test))

SVM_avg = sum(SVM)/len(SVM)
KNN_avg = sum(KNN)/len(KNN)
LR_avg = sum(LR)/len(LR)

print('SVM', SVM_avg)

print('Logistic Regression', LR_avg)

print('K-Nearest Neighbour', KNN_avg)

out_pred = neigh.predict(in_test)

#Going for Confusion Matrix
conf_matrix = confusion_matrix(out_test, out_pred)
print(conf_matrix)
fig = plt.figure()
axs = fig.add_subplot(111)
caxs = axs.matshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(caxs)
Activities = ["Climbing", "Walking", "Jumping", "Running"]
axs.set_xticklabels([""] + Activities)
axs.set_yticklabels([""] + Activities)
plt.show()
