import joblib
model = joblib.load('knn_model.sav')

test=[0.4,2.1,5.1,2.2],[2.1,5.5,2.1,1.1]


result = model.predict(test)
print('Predicted Result:', result)