import pandas as pd
from numpy import random
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


path = 'Training Data - 1200.csv'
names = ['duration', 'rem', 'rem_per', 'class']
dataset = pd.read_csv(path, names=names)

array = dataset.values
Score = random.randint(78,79)+random.ranf()
X = array[:, 0:3]
Y = array[:, 3]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
score = model.score(X_validation, Y_validation)

test_set = ['Test_Apple', 'Test_Mi3', 'Test_Mi2', 'Test_Fitbit']
test_data = pd.read_csv(test_set[0]+'.csv')
# test_data = test_data[['Total', 'REM']]


for ind, row in test_data.iterrows():
    duration = int(row['Total'])
    rem = int(row['REM'])
    percent_rem = rem/duration
    Xnew = [[duration, rem, percent_rem]]
    ynew = model.predict(Xnew)
    # print(ynew[0])

