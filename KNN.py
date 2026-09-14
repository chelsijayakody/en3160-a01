import pandas as pd

dataset = pd.read_csv(r'C:\Users\HP\Downloads\iris.csv').values

print(dataset,dataset.shape)

data = dataset[:, 0:4]
print(data,data.shape)

target = dataset[:, 4]
print(target,target.shape)

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)

print(train_data, train_data.shape)
print(test_data, test_data.shape)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

model.fit(train_data, train_target) #training the model

predicted_target = model.predict(test_data) #testing the model
print('Predicted Target:', predicted_target)

print('Test Target:', test_target)

from sklearn.metrics import accuracy_score
acc = accuracy_score(test_target, predicted_target)
print('Test Accuracy:', acc)

predicted_target = model.predict(train_data) #testing the model
acc= accuracy_score(train_target, predicted_target)
print('Training Accuracy:', acc)

import joblib

# Save the trained model
joblib.dump(model, 'knn_model.sav')
