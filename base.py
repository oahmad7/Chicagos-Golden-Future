import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3 as sql
import pandas as pd
import os
import os.path
import time
import traceback
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


import gmaps
import requests
import urllib
import warnings
import random

#Get API Key
GOOGLE_MAPS_API_KEY = -1
with open('apikey.txt') as f:
    GOOGLE_MAPS_API_KEY = f.readline()
    f.close
    
GOOGLE_MAPS_API_KEY = str(GOOGLE_MAPS_API_KEY)

# Functions needed to process a csv into a sqlite database
# This helps us get past the low memory errors that pandas gives us since our data is so large
def process(columns,columnString, valueString, chunk,cur):
    cur.executemany("INSERT INTO data VALUES (" + valueString +")", chunk.filter(items=columns).values.tolist())

def toDB(dbName, fileName,columns):

    #check if file exists
    PATH='./' + dbName

    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print ("Database " + dbName + " already exists. Stopping.")
        return
    
    #open the connection to our db
    conn = sql.connect(dbName)
    cur = conn.cursor()
    
    #Get header
    numfields = pd.read_csv(fileName, delimiter='\t',nrows=1,skipinitialspace=1,encoding="utf-8-sig")

    #Build dynamic string for database query
    columnString = ""
    valueString = ""
    for column in columns:
        fixedC = column.replace(" ","_").replace(".","")
        columnString += fixedC + ", "
        valueString += "?, "
    columnString = columnString[0:-2].strip()
    valueString = valueString[0:-2].strip()

    #Create table
    try:
        cur.execute("Create table data (" + columnString + ")")
    except Exception: 
        traceback.print_exc()
        print ("Database " + dbName + " already exists. Stopping.")
        return
    

    print("Processing File: " + fileName)
    chunksize = 10 ** 4
    for chunk in pd.read_csv(fileName, chunksize=chunksize, delimiter='\t',skipinitialspace=1,encoding="utf-8-sig",low_memory=False):
        process(columns,columnString,valueString,chunk,cur)

    #close the db connection
    print("Done processing, now writing to database: " + dbName)
    conn.commit()
    cur.close()
    conn.close()

def makeDB():
    # Now we build the databases, while doing a bit of data cleaning in the process.
    # First we need to list out what column names we want to keep
    # This will build a table around those columns and insert the corresponding values into the database

    #handle building permit data
    dbColumns = ['ID','PERMIT_TYPE','ISSUE_DATE','ESTIMATED_COST','TOTAL_FEE','WORK_DESCRIPTION','LATITUDE','LONGITUDE','LOCATION']
    toDB("Building_Permits.db", "Building_Permits.tsv", dbColumns)

    #handle city land data
    dbColumns = ['ID','Property Status','Date of Acquisition','Date of Disposition','Sq. Ft.','Ward','Community Area Number','Community Area Name','Zip Code','X Coordinate','Y Coordinate','Latitude','Longitude','Location']
    toDB("City-Owned_Land_Inventory.db", "City-Owned_Land_Inventory.tsv",dbColumns)

    #handle crimes data (This will take a while)
    dbColumns = ['ID', 'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description','Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward','Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year', 'Updated On','Latitude', 'Longitude', 'Location']
    toDB("Crimes_-_2001_to_present.db", "Crimes_-_2001_to_present.tsv",dbColumns)

def getZipcodeFromLatLon(lat,lon):
    #Encode parameters 
    params = urllib.parse.urlencode({'latitude': lat, 'longitude':lon, 'format':'json'})
    #Contruct request URL
    url = 'https://maps.googleapis.com/maps/api/geocode/json?latlng='+ str(lat) + ',' + str(lon) + '&sensor=true&key=' + GOOGLE_MAPS_API_KEY


    #Get response from API
    response = requests.get(url)

    #Parse json in response
    data = response.json()

    locData = data['results'][0]['address_components']

    zipcode = -1
    for loc in locData:
        if(loc['types'][0] == 'postal_code'):
            zipcode = loc['long_name']
    return zipcode

def getYears(db):
    conn = sql.connect(db)
    cur = conn.cursor()

    #select all data for chicago
    cur.execute("SELECT DISTINCT YEAR FROM data ORDER BY Year ASC")

    #Get all values
    crimeCount = cur.fetchall()

    #Cleanup db connection
    cur.close()
    conn.close()
    
    years = []
    
    for c in crimeCount:
        years.append(c[0])
    
    return years
    

def crimesVis():
    #Get connection to crimes database
    conn = sql.connect("Crimes_-_2001_to_present.db")
    cur = conn.cursor()

    #select all data for chicago
    cur.execute("SELECT COUNT(*) as count FROM data WHERE Year >= 2006 GROUP BY Year ORDER BY Year ASC")

    #Get all values
    crimeCount = cur.fetchall()

    #Cleanup db connection
    cur.close()
    conn.close()

    #Build visualization
    years = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    countList = [count[0] for count in crimeCount]
    plt.bar(years, countList)
    ax = plt.gca()
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    plt.title('Count of Crimes by Year')
    plt.show()

def permitVis():
    #Get connection to building permits database
    conn = sql.connect("Building_Permits.db")
    cur = conn.cursor()

    #Get distinct counts for types of permits
    cur.execute("SELECT PERMIT_TYPE,COUNT(*) as count FROM data GROUP BY PERMIT_TYPE ORDER BY count DESC")

    #Get all values
    permitTypes = cur.fetchall()

    #Cleanup db connection
    cur.close()
    conn.close()

    permits = []
    permitsCount = []

    for permit in permitTypes:
        permits.append(permit[0].replace("PERMIT - ",""))
        permitsCount.append(permit[1])

    # #Build visualization
    ax = plt.gca()
    plt.figure(1, [20, 8])
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    plt.bar(permits,permitsCount)
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.title('Permit Types In Chicago')

    # Set the x-axis limit
    # Change of fontsize and angle of xticklabels

    plt.show()

def heatmapVis():
    #Get connection to crimes database
    conn = sql.connect("Crimes_-_2001_to_present.db")
    cur = conn.cursor()

    #select all data for chicago
    cur.execute("SELECT Latitude, Longitude FROM data WHERE Year >= 2006 AND Latitude IS NOT NULL AND Longitude IS NOT NULL")

    #Get all values
    crimeLocation = cur.fetchall()

    #Cleanup db connection
    cur.close()
    conn.close()

    #randomly shuffles the crime location to remove bias when graphing in map
    np.random.seed(100)
    np.random.shuffle(crimeLocation)

    locations=[]
    for i in range(0,30000,2):
        locations.append(crimeLocation[i])

    #Get connection to crimes database
    conn = sql.connect("City-Owned_Land_Inventory.db")
    cur = conn.cursor()

    #select all data for chicago
    cur.execute("SELECT Latitude, Longitude FROM data WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL")

    #Get all values
    landLocation = cur.fetchall()

    #Cleanup db connection
    cur.close()
    conn.close()

    #randomly shuffles the crime location to remove bias when graphing in map
    np.random.seed(100)
    np.random.shuffle(landLocation)

    landlocations=[]
    for i in range(0,len(landLocation),2):
        landlocations.append(landLocation[i])

    gmaps.configure(api_key=GOOGLE_MAPS_API_KEY)

    #centers map on Illinois
    illinoisLocation = (41.881832, -87.623177)
    fig = gmaps.figure(center=illinoisLocation, zoom_level=10)

    #creates heat map from crime locations
    crimeHeatMap = gmaps.heatmap_layer(locations)

    #creates marker from land location
    landSymbol = gmaps.symbol_layer(landlocations, fill_color="rgba(0,0,200,0.4)", stroke_color="rgba(0,0,200,0.4)", scale=2)

    #plots everything on figure
    fig.add_layer(crimeHeatMap)
    fig.add_layer(landSymbol)
    return fig


def zipcodeDF():

    #Get connection to building permits database
    conn = sql.connect("Building_Permits.db")
    cur = conn.cursor()

    #Get distinct counts for types of permits
    cur.execute("SELECT * FROM data WHERE (substr(ISSUE_DATE,7,4) = '2018') and Latitude IS NOT NULL AND Longitude IS NOT NULL")


    #Get all values
    permits = cur.fetchall()
    #Cleanup db connection
    cur.close()
    conn.close()

    #Make a dataframe for it now
    df = pd.DataFrame(permits)
    df.columns = ['ID','PERMIT_TYPE','ISSUE_DATE','ESTIMATED_COST','TOTAL_FEE','WORK_DESCRIPTION','LATITUDE','LONGITUDE','LOCATION']

    #This is going to take quite a while (approx 6 hours for 1 year of data)
    #For each of these rows, find a matching zipcode
    # for permit in range(0,len(permits)):
    #     zipcode = getZipcodeFromLatLon(permits[permit][6],

    #                                    permits[permit][7])
    #     zipcodes.append(zipcode)

    # df.insert(6, "ZIPCODE", zipcodes, True)
    # out_filepath = "permitsWithZipcodes.csv"
    # df.to_csv(out_filepath, index=False)

    #Now that we have all of our zipcode data added to the permits, lets make some visualizations for it
    zips = pd.read_csv('permitsWithZipcodes.csv',header=0)
    zips.head()
    zips['COUNTER'] =1       #initially, set that counter to 1.
    groupedZipcodes = zips.groupby(['ZIPCODE','PERMIT_TYPE'])['ESTIMATED_COST','COUNTER'].sum().reset_index() #sum function
    groupedZipcodes.columns = ["zipcode",'permit_type',"ESTIMATED_COST","count"]
    warnings.filterwarnings('ignore', 'You have mixed*')

    return groupedZipcodes
    
def zipcodeVis(df, zipcode):
    labels = []
    counts = []
    currentZip = 60018
    total = 0
    totalInvest = 0
    for index, row in df.iterrows():
        if(row['zipcode'] != -1 and row['zipcode'] != 21234):
            #If we are still in the same zipcode range
            if(currentZip == row['zipcode']):
                labels.append(row['permit_type'].replace("PERMIT - ",""))
                counts.append(row['count'])
                total += row['count']
                totalInvest += row['ESTIMATED_COST']
            else:
                if(currentZip == zipcode ):
                    print("Zipcode: " + str(currentZip) + "\nTotal Permits: " + str(total))
                    #We are in a new zipcode so show the previous plot
                    counts, labels = zip(*sorted(zip(counts, labels)))
                    
                    
                    patches, texts, _ = plt.pie(counts[::-1], autopct='',
                                                startangle=0)
                    plt.legend(patches,
                               bbox_to_anchor=(1,1),
                               borderpad=0,
                               labelspacing=2,
                               frameon=False,
                               labels=['%s   (%1.1f %%)' % (l, s/total * 100) for l, s in zip(labels[::-1], counts[::-1])])
                    #draw circle
                    centre_circle = plt.Circle((0,0),0.70,fc='white')
                    fig = plt.gcf()
                    fig.gca().add_artist(centre_circle)
                    
                    plt.show()
                
                
                #Reset the previous
                currentZip = row['zipcode']
                labels = []
                counts = []
                total = row['count']
                totalInvest = row['ESTIMATED_COST']
                labels.append(row['permit_type'].replace("PERMIT - ",""))
                counts.append(row['count'])


def getMLTable():
    conn = sql.connect("Building_Permits.db")
    cur = conn.cursor()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - ELECTRIC WIRING' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    ewht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - EASY PERMIT PROCESS' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    eppht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - RENOVATIONAL/ALTERATION' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    rht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - SIGNS' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    siht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - NEW CONSTRUCTION' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    ncht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - WRECKING/DEMOLITION' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    wht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - ELEVATOR EQUIPMENT' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    eeht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - SCAFFOLDING' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    sht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - PORCH CONSTRUCTION' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    pcht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - REINSTATE REVOKED PMT' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    rrpht = cur.fetchall()

    cur.execute("SELECT CASE WHEN PERMIT_TYPE = 'PERMIT - FOR EXTENSION OF PMT' THEN 1 ELSE 0 END FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    fepht = cur.fetchall()

    cur.execute("SELECT ESTIMATED_COST, TOTAL_FEE FROM data WHERE substr(ISSUE_DATE,7,4) = '2018' AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    permit = cur.fetchall()

    #Cleanup db connection
    cur.close()
    conn.close()

    conn = sql.connect("City-Owned_Land_Inventory.db")
    cur = conn.cursor()

    cur.execute("SELECT Zip_Code, Ward, Property_Status, Sq_Ft FROM data WHERE Zip_Code IS NOT 0 AND Latitude IS NOT NULL AND Longitude IS NOT NULL")
    land = cur.fetchall()

    #Cleanup db connection
    cur.close()
    conn.close()

    conn = sql.connect("Crimes_-_2001_to_present.db")
    cur = conn.cursor()

    cur.execute("SELECT Ward, Primary_Type, Arrest FROM data WHERE Year = 2018 AND Latitude IS NOT NULL AND Longitude IS NOT NULL AND Primary_Type IS NOT NULL")
    crimes = cur.fetchall()

    #Cleanup db connection
    cur.close()
    conn.close()

    permitData = {'ELECTRIC WIRING':[X[0] for X in ewht], 'EASY PERMIT PROCESS':[X[0] for X in eppht], 'RENOVATIONAL/ALTERATION':[X[0] for X in rht], 'SIGNS':[X[0] for X in siht], 'NEW CONSTRUCTION':[X[0] for X in ncht],
        'WRECKING/DEMOLITION':[X[0] for X in wht], 'ELEVATOR EQUIPMENT':[X[0] for X in eeht], 'SCAFFOLDING':[X[0] for X in sht], 'PORCH CONSTRUCTION':[X[0] for X in pcht], 'REINSTATE REVOKED PMT':[X[0] for X in rrpht],
        'FOR EXTENSION OF PMT':[X[0] for X in fepht], "ZIPCODE": pd.read_csv("permitsWithZipcodes.csv")['ZIPCODE'], "ESTIMATED_COST":[X[0] for X in permit], "TOTAL_FEE":[X[1] for X in permit]}

    binStatus = []
    for i in land:
        if i[2] == "Sold":
            binStatus.append(1)
        else:
            binStatus.append(0)

    cityData = {"ZIPCODE" : [X[0] for X in land], 'WARD': [X[1] for X in land], "SQ_FT": [X[3] for X in land], 'STATUS': binStatus}

    crimeToNum = []
    for i in crimes:
        if i[1] == "BATTERY":
            crimeToNum.append(1)
        elif i[1] == "ASSAULT":
            crimeToNum.append(2)
        elif i[1] == "OTHER OFFENSE":
            crimeToNum.append(3)
        elif i[1] == "THEFT":
            crimeToNum.append(4)
        elif i[1] == "NARCOTICS":
            crimeToNum.append(5)
        elif i[1] == "PUBLIC PEACE VIOLATION":
            crimeToNum.append(6)
        elif i[1] == "CRIMINAL DAMAGE":
            crimeToNum.append(7)
        elif i[1] == "CRIMINAL TRESPASS":
            crimeToNum.append(8)
        elif i[1] == "MOTOR VEHICLE THEFT":
            crimeToNum.append(9)
        elif i[1] == "DECEPTIVE PRACTICE":
            crimeToNum.append(10)
        elif i[1] == "INTERFERENCE WITH PUBLIC OFFICER":
            crimeToNum.append(11)
        elif i[1] == "ROBBERY":
            crimeToNum.append(12)
        elif i[1] == "BURGLARY":
            crimeToNum.append(13)
        elif i[1] == "LIQUOR LAW VIOLATION":
            crimeToNum.append(14)
        elif i[1] == "WEAPONS VIOLATION":
            crimeToNum.append(15)
        elif i[1] == "SEX OFFENSE":
            crimeToNum.append(16)
        elif i[1] == "OFFENSE INVOLVING CHILDREN":
            crimeToNum.append(17)
        elif i[1] == "KIDNAPPING":
            crimeToNum.append(18)
        elif i[1] == "CRIM SEXUAL ASSAULT":
            crimeToNum.append(19)
        elif i[1] == "GAMBLING":
            crimeToNum.append(20)
        elif i[1] == "ARSON":
            crimeToNum.append(21)
        elif i[1] == "PROSTITUTION":
            crimeToNum.append(22)
        elif i[1] == "STALKING":
            crimeToNum.append(23)
        elif i[1] == "OBSCENITY":
            crimeToNum.append(24)
        elif i[1] == "CONCEALED CARRY LICENSE VIOLATION":
            crimeToNum.append(25)
        elif i[1] == "HUMAN TRAFFICKING":
            crimeToNum.append(26)
        elif i[1] == "NON-CRIMINAL":
            crimeToNum.append(27)
        elif i[1] == "HOMICIDE":
            crimeToNum.append(28)
        elif i[1] == "PUBLIC INDECENCY":
            crimeToNum.append(29)
        elif i[1] == "NON-CRIMINAL (SUBJECT SPECIFIED)":
            crimeToNum.append(30)
        elif i[1] == "OTHER NARCOTIC VIOLATION":
            crimeToNum.append(31)
        elif i[1] == "INTIMIDATION":
            crimeToNum.append(32)

    crimeData = {"WARD":[X[0] for X in crimes], "ARREST":[X[2] for X in crimes], "Crime_Type": crimeToNum}

    permitDF = pd.DataFrame(permitData)
    cityDF = pd.DataFrame(cityData)
    crimeDF = pd.DataFrame(crimeData)

    permit_cityDF = pd.merge(permitDF, cityDF, how='inner', left_on = 'ZIPCODE', right_on = 'ZIPCODE').dropna().reset_index(drop=True)

    ht1=[]
    ht2=[]
    ht3=[]
    ht4=[]
    ht5=[]
    ht6=[]
    ht7=[]
    ht8=[]
    ht9=[]
    ht10=[]
    ht11=[]
    zip_=[]
    estcost=[]
    fee=[]
    ward=[]
    sf=[]
    stat=[]
    for i in range(0, 20000):
        randNum = random.randint(1, permit_cityDF.shape[0])
        ht1.append(permit_cityDF.at[randNum,'ELECTRIC WIRING'])
        ht2.append(permit_cityDF.at[randNum,'EASY PERMIT PROCESS'])
        ht3.append(permit_cityDF.at[randNum,'RENOVATIONAL/ALTERATION'])
        ht4.append(permit_cityDF.at[randNum,'SIGNS'])
        ht5.append(permit_cityDF.at[randNum,'NEW CONSTRUCTION'])
        ht6.append(permit_cityDF.at[randNum,'WRECKING/DEMOLITION'])
        ht7.append(permit_cityDF.at[randNum,'ELEVATOR EQUIPMENT'])
        ht8.append(permit_cityDF.at[randNum,'SCAFFOLDING'])
        ht9.append(permit_cityDF.at[randNum,'PORCH CONSTRUCTION'])
        ht10.append(permit_cityDF.at[randNum,'REINSTATE REVOKED PMT'])
        ht11.append(permit_cityDF.at[randNum,'FOR EXTENSION OF PMT'])
        zip_.append(permit_cityDF.at[randNum,'ZIPCODE'])
        fee.append(permit_cityDF.at[randNum,'TOTAL_FEE'])
        estcost.append(permit_cityDF.at[randNum,'ESTIMATED_COST'])
        ward.append(permit_cityDF.at[randNum,'WARD'])
        sf.append(permit_cityDF.at[randNum,'SQ_FT'])
        stat.append(permit_cityDF.at[randNum,'STATUS'])


    finData = {'ELECTRIC WIRING':ht1, 'EASY PERMIT PROCESS':ht2, 'RENOVATIONAL/ALTERATION':ht3, 'SIGNS':ht4, 'NEW CONSTRUCTION':ht5,
        'WRECKING/DEMOLITION':ht6, 'ELEVATOR EQUIPMENT':ht7, 'SCAFFOLDING':ht8, 'PORCH CONSTRUCTION':ht9, 'REINSTATE REVOKED PMT':ht10,
            'FOR EXTENSION OF PMT':ht11, "ZIPCODE": zip_, "ESTIMATED_COST":estcost, "TOTAL_FEE":fee, 'WARD': ward, "SQ_FT": sf, 'STATUS': stat}
    finDF = pd.DataFrame(finData)

    joinedDF = pd.merge(crimeDF, finDF, how='inner', left_on = 'WARD', right_on = 'WARD').dropna().reset_index(drop=True)
    return joinedDF

def mlSplit(data,ammount):
    np.random.seed(100)
    #randomize
    shuffled = data.sample(frac=1)
    features = []
    status = []
    counter = 0
    testCount = int(shuffled.shape[0] * ammount)
    sample = data.head(testCount)
    for index, row in sample.iterrows():
        features.append(row[:15])
        status.append(str(row[16]))
    return features,status

def runNeuralNet(X,Y):
    layer = [1,2,5]
    node = [5,10,25,50,100]
    minCVError = 1
    minNodes = 0
    minLayers = 0
    for layers in layer:
        for nodes in node:
            layers2 =()
            for k in range(0,layers):
                layers2 += nodes,
        
            start = time.time()
            nn = MLPClassifier(hidden_layer_sizes=layers2, max_iter=10000, activation='relu', solver='adam', epsilon=0.001, alpha=0)
            nnModel = nn.fit(X, Y)
            end = time.time()
            
            start = time.time()
            cv2 = 1 - sk.cross_val_score(nnModel, X, Y, cv=10, scoring='accuracy').mean()
            end = time.time()
            
            if(minCVError > cv2):
                minCVError = cv2
                minNodes = nodes
                minLayers = layers

    print("Min CV Error is ", minCVError, ' with ', minNodes, ' nodes and ', minLayers, ' layers')

def runSVM(X,Y):
    
    c = [0.001, 0.01, 0.1, 1, 10, 100]

    cv = []
    minCV = 1
    minC = 0
    for i in c:
        n_estimators = 10

        start = time.time()
        clf = OneVsRestClassifier(BaggingClassifier(SVC(C=i, kernel='rbf', gamma='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
        svmFit = clf.fit(X, Y)
        end = time.time()

        start = time.time()
        cverror = 1-cross_val_score(svmFit, X, Y, cv=10, scoring='accuracy').mean()
        end = time.time()

        if minCV > cverror:
            minCV = cverror
            minC=i

        cv.append(cverror)

    plt.scatter(c, cv, s=10)
    plt.title("SVM Cross Validation Errors")
    plt.xlabel("C")
    plt.ylabel("CV Error")
    plt.show()

    print("Lowest cross validation error was ", minCV, " at C value ", minC)
    #[0.6174594213020366, 0.5987514023980376, 0.5738948174372289, 0.5827760892976669]
