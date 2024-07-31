import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
data = music_data.drop(columns=['genre'])
genres = music_data['genre']
data_train, data_test, genres_train, genres_test = train_test_split(data, genres, test_size=0.2)
# print(music_data)

model = DecisionTreeClassifier()
model.fit(data_train, genres_train)
predictions = model.predict(data_test)

score = accuracy_score(genres_test, predictions)
print(score)