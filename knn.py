import numpy as np 
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_csv("data.csv")

#editing the dataset for future use
# data = data[["overall", "potential", "player_positions", "international_reputation", "pace", "shooting", "passing", "dribbling", "defending", "physic", "gk_diving","gk_handling","gk_kicking","gk_reflexes","gk_speed","gk_positioning","attacking_crossing","attacking_finishing","attacking_heading_accuracy","attacking_short_passing","attacking_volleys","skill_dribbling","skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control","movement_acceleration","movement_sprint_speed","movement_agility","movement_reactions","movement_balance","power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots","mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision","mentality_penalties","mentality_composure","defending_marking","defending_standing_tackle","defending_sliding_tackle","goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking","goalkeeping_positioning","goalkeeping_reflexes"]]
# data.to_csv(r'data.csv', index=False)

le = preprocessing.LabelEncoder()
data.fillna(0) #NaN -> 0

original_values = data.overall.unique() #used later
data = data.apply(le.fit_transform)

X = data.drop("overall", 1).to_numpy() 
Y = data["overall"].to_numpy()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.05) #0.05 for test data due to the size of the dataset (over 18k records)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.05)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

#testing what number of neighbours works the best with this dataset
best_k = 3
best_score = acc

for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.05)
    for k in range(4, 12):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)
        if acc > best_score:
            best_score = acc
            best_k = k

print("best fitting k: ", best_k, " with accuracy of ",best_score)

"""
the loop above usually indicates that the best fitting k is k=5 (~0.956 acc)
"""

from joblib import load, dump

best_score = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.05) 

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    if acc > best_score:
        best_score = acc
        dump(model, "trained")
        print("new best score: ",acc)


model = load("trained")

#finding out what range of overall player scores gets predicted the best
import matplotlib.pyplot as plt

pred_encoded = model.predict(x_test)

actual = [original_values[len(original_values)-1-x] for x in y_test]
predictions = [original_values[len(original_values)-1-x] for x in pred_encoded]
plt.plot(actual, predictions, 'bo')
plt.xlabel("actual values")
plt.ylabel("predicted values")
plt.show()