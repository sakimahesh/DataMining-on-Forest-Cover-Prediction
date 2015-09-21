import pandas as pd
from sklearn import ensemble
from sklearn import cross_validation

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#Shifting Aspect at around 180 degrees range

def angular(x):
    if x + 180 > 360:
        return (x-180)
    else:
        return (x + 180)

train['Aspect2'] = train.Aspect.map(angular)
test['Aspect2'] = test.Aspect.map(angular)

#Vertical_Distance_To_Hydrology have some negative values so creating variable indicating positive or negative values

train['Highwater'] = train.Vertical_Distance_To_Hydrology < 0
test['Highwater'] = test.Vertical_Distance_To_Hydrology < 0

train['EVDtH'] = train.Elevation - train.Vertical_Distance_To_Hydrology
test['EVDtH'] = test.Elevation - test.Vertical_Distance_To_Hydrology

train['EHDtH'] = train.Elevation - train.Horizontal_Distance_To_Hydrology * 0.2
test['EHDtH'] = test.Elevation - test.Horizontal_Distance_To_Hydrology * 0.2


# Creating similar distance measures by playing around with distances
train['Distanse_to_Hydrolody'] = (train['Horizontal_Distance_To_Hydrology'] ** 2 + train[
    'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
test['Distanse_to_Hydrolody'] = (test['Horizontal_Distance_To_Hydrology'] ** 2 + test[
    'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5

train['Hydro_Fire_1'] = train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Fire_Points']
test['Hydro_Fire_1'] = test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Fire_Points']

train['Hydro_Fire_2'] = abs(train['Horizontal_Distance_To_Hydrology'] - train['Horizontal_Distance_To_Fire_Points'])
test['Hydro_Fire_2'] = abs(test['Horizontal_Distance_To_Hydrology'] - test['Horizontal_Distance_To_Fire_Points'])

train['Hydro_Road_1'] = abs(train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_1'] = abs(test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways'])

train['Hydro_Road_2'] = abs(train['Horizontal_Distance_To_Hydrology'] - train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_2'] = abs(test['Horizontal_Distance_To_Hydrology'] - test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_1'] = abs(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_1'] = abs(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_2'] = abs(train['Horizontal_Distance_To_Fire_Points'] - train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_2'] = abs(test['Horizontal_Distance_To_Fire_Points'] - test['Horizontal_Distance_To_Roadways'])


# Performing Classification on the feature Set

feature_cols = [col for col in train.columns if col not in ['Cover_Type','Id']]
X_train = train[feature_cols]
X_test = test[feature_cols]
y = train['Cover_Type']
test_ids = test['Id']

forest=ensemble.ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0)
scores = cross_validation.cross_val_score(forest, X_train, y, cv=5)
print "Mean Accuracy", scores.mean()
forest.fit(X_train, y)
Y_test = forest.predict(X_test)

'''
X,Xt,y,yt = cross_validation.train_test_split(trainX,trainy,test_size=0.3)
randfrst.fit(X,y)
y_rf = randfrst.predict(Xt)
print metrics.classification_report(yt,y_rf)
print metrics.accuracy_score(yt,y_rf)
randfrst.fit(trainX,trainy)
y_test_rf = randfrst.predict(testX)
'''

print("Creating csv file")
pd.DataFrame({'Id':test.Id.values,'Cover_Type':Y_test}).sort_index(ascending=False,axis=1).to_csv('ans.csv',index=False)
print("Done Predicting Forest Cover")
