import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")
Window_Length = 100

df = pd.read_csv("Acc_Gyro_F.csv")
test_input = []
test_output = []

i = 0
while(i <= df.shape[0] - Window_Length):
    temp = df[i:i+Window_Length]
    inp = temp.drop(temp[['Output']],axis = 1)
    out = temp[['Output']]
    test_input.append(inp)
    test_output.append(out.iloc[0])
    i = i + int(Window_Length/2)

mean = pd.DataFrame()
standard_deviation = pd.DataFrame()
mad = pd.DataFrame()
avgdiff = pd.DataFrame()
maximum = pd.DataFrame()
minimum = pd.DataFrame()

for windows in test_input:
    df_mean = pd.DataFrame(windows.mean()).transpose()
    df_std = pd.DataFrame(windows.std()).transpose()
    df_max = pd.DataFrame(windows.max()).transpose()
    df_med_abs = pd.DataFrame(windows.mad()).transpose()
    df_avg_diff = pd.DataFrame(windows.diff().dropna().div(10).mean()).transpose()
    df_min = pd.DataFrame(windows.min()).transpose()
    mean = pd.concat([mean, df_mean])
    standard_deviation = pd.concat([standard_deviation, df_std])
    mad = pd.concat([mad, df_med_abs])
    maximum = pd.concat([maximum, df_max])
    minimum = pd.concat([minimum, df_min])
    avgdiff = pd.concat([avgdiff, df_avg_diff])

mean = mean.reset_index(drop=True)
standard_deviation = standard_deviation.reset_index(drop=True)
mad = mad.reset_index(drop=True)
avgdiff = avgdiff.reset_index(drop=True)
maximum = maximum.reset_index(drop=True)
minimum = minimum.reset_index(drop=True)

mean.columns = [x + "mean" for x in mean.columns]
standard_deviation.columns = [x + "std" for x in standard_deviation.columns]
mad.columns = [x + "mad" for x in mad.columns]
avgdiff.columns = [x + "avgdiff" for x in avgdiff.columns]
maximum.columns = [x + "max" for x in maximum.columns]
minimum.columns = [x + "min" for x in minimum.columns]

feature_columns = mean.join([standard_deviation, mad, avgdiff, maximum, minimum])
print(feature_columns)

feature_columns['Output'] = np.array(test_output)

train, test = train_test_split(feature_columns, test_size=0.25, shuffle=True)
in_train = train.drop(train[['Output']],axis=1)
out_train = train[['Output']]

in_test = test.drop(test[['Output']],axis=1)
out_test = test[['Output']]

scaler = StandardScaler()
scaler.fit(in_train)
in_train = scaler.transform(in_train)
in_test = scaler.transform(in_test)

#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(x_train, y_train)
#y_pred = neigh.predict(x_test)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(in_train, out_train)

#clf.predict(x_train, y_train)

print(clf.score(in_test, out_test))


#print(neigh.score(x_test, y_test))
#print(df.shape)
