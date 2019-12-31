import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymysql
from flask import Flask, render_template
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from sklearn.neural_network import MLPClassifier

connection = pymysql.connect(host='localhost', user='root', password='666', db='crimes')
connection2 = pymysql.connect(host='localhost', user='root', password='666', db='city_lands')


app = Flask(__name__, template_folder=".")
GoogleMaps(app)


#aggregate using count, grouping by year
#purpose is to grph bar plot with y being count and x being Year
'''
with connection:
    
    cur = connection.cursor()
    cur.execute("SELECT COUNT(*) as count FROM crimes WHERE Year >= 2006 GROUP BY Year ORDER BY Year ASC")
    
    crimeCount = cur.fetchall()

cur.close()
connection.close()

years = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
countList = [count[0] for count in crimeCount]
plt.bar(years, countList)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Count of crimes throughtout years')
plt.show()
'''
with connection:
    cur = connection.cursor()
    cur.execute("SELECT Latitude, Longitude FROM crimes WHERE Year >= 2006 AND Latitude IS NOT NULL AND Longitude IS NOT NULL AND Primary_Type IS NOT NULL")
    
    crimeLocation = cur.fetchall()

cur.close()

with connection:
    cur = connection.cursor()
    cur.execute("SELECT Primary_Type FROM crimes WHERE Year >= 2006 AND Latitude IS NOT NULL AND Longitude IS NOT NULL AND Primary_Type IS NOT NULL")
    
    crimeType = cur.fetchall()

cur.close()
connection.close()

X_train, X_test, Y_train, Y_test = sk.train_test_split(crimeLocation, crimeType, test_size=0.9, train_size=0.1)


xTrain = [[X[0],X[1]] for X in X_train]
yTrain = [Y[0] for Y in Y_train]


#k = [1,2,3,4,5,6,7,8,9,10]


#cverrore = []
#cverrorm = []
#cverrorc = []


print('lets begin')

layer = [1,2,5]
node = [2,5,10,50]
layers2=()


minCVError = 1
minNodes = 0
minLayers = 0
for layers in layer:
    for nodes in node:
        for k in range(0,layers):
                layers2 += nodes,

        nn = MLPClassifier(hidden_layer_sizes=layers2, max_iter=1000, activation='relu', solver='adam', epsilon=0.001, alpha=0)
        nnModel = nn.fit(xTrain, yTrain)
        cv2 = 1 - sk.cross_val_score(nnModel, xTrain, yTrain, cv=10, scoring='accuracy').mean()

        if(minCVError > cv2):
            minCVError = cv2
            minNodes = nodes
            minLayers = layers
            
        layers2=()

print("Min CV Error is ", minCVError, ' with ', minNodes, ' nodes and ', minLayers, ' layers')



'''
    if cve < minCv:
        minCv = cve
        minNeighbor = i
        metric = 'euclidean'
        
    if cvm < minCv:
        minCv = cvm
        minNeighbor = i
        metric = 'manhattan'
        
    if cvc < minCv:
        minCv = cvc
        minNeighbor = i
        metric = 'chebyshev'
    print(i)


print('Min cross validation error: ', minCv, ' with neighbors: ', minNeighbor,' and metric ', metric)

eu = plt.scatter(k, cverrore, color='red', label='euclidean')
man = plt.scatter(k, cverrorm, color='green', label='manhattan')
cheb = plt.scatter(k, cverrorc, color='blue', label='chebyshev')

plt.legend(handles=[eu, man, cheb])
plt.xlabel('K-Neighbors')
plt.ylabel('Cross Validation Error')
plt.title('CV error vs KNN')
plt.show()
'''
'''
with connection2:
    
    cur2 = connection2.cursor()
    cur2.execute("SELECT Latitude, Longitude FROM city_lands WHERE Property_Status = 'Owned by City' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")

    land = cur2.fetchall()
cur2.close()
connection2.close()


@app.route("/")
def mapview():
    mymap = Map(
        identifier="crimemap",
        lat=land[0][0],
        lng=land[0][1],
        markers=[(lands[0], lands[1]) for lands in land],
        markers =[(crimes[0], crimes[1]) for crimes in crime]
    )
    return render_template('example.html', mymap=mymap)

if __name__ == "__main__":
    app.run(debug=True)
'''

