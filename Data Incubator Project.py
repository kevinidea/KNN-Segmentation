import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.cluster.vq import vq

def importData(path, fileName):
    dataset = pd.read_csv(path + '/'+fileName)
    return dataset
    
#import all 3 USA datasets 
dataSet1 = importData("D:/Documents/Big Data/The Data Incubator/Potential Donors", 'eo1.csv')
dataSet2 = importData("D:/Documents/Big Data/The Data Incubator/Potential Donors", 'eo2.csv')
dataSet3 = importData("D:/Documents/Big Data/The Data Incubator/Potential Donors", 'eo3.csv')

#consolidate all 4 datasets into 1 dataset
allData =pd.concat([dataSet1,dataSet2, dataSet3])

#explore data
print(allData.describe())

#pick the features for unsupervised machine learning (clustering)
assetAndIncome = allData.loc[:,['STATE','ASSET_AMT', 'INCOME_AMT']]

#fill all the missing data with 0 
df = assetAndIncome.fillna(value = 0)

#select only the records with either positive asset OR income amount
positiveAssetOrIncome = df[(df['ASSET_AMT'] >  0) | (df['INCOME_AMT'] >  0)]

#examine positiveAssetOrIncome data
#fig1 = plt.figure()
#plt.scatter(x=positiveAssetOrIncome.loc[:,'ASSET_AMT'], y= positiveAssetOrIncome.loc[:,'INCOME_AMT'])
print(positiveAssetOrIncome.describe())

#remove outliners outside of 2 standard deviations
assetStd = np.std(positiveAssetOrIncome.loc[:,'ASSET_AMT'])
incomeStd = np.std(positiveAssetOrIncome.loc[:,'INCOME_AMT'])
positiveAssetOrIncome2 = positiveAssetOrIncome[(positiveAssetOrIncome['ASSET_AMT'] < 2*assetStd) & (positiveAssetOrIncome['INCOME_AMT'] < 2*incomeStd ) ]

#examine 2nd positiveAssetOrIncome after removing outliners
#fig2 = plt.figure()
#plt.scatter(x= positiveAssetOrIncome2.loc[:,'ASSET_AMT'], y = positiveAssetOrIncome2.loc[:,'INCOME_AMT'])
print (positiveAssetOrIncome2.describe())

#convert Pandas dataframe to Numpy ndarray 2dMatrix for machine learning algorithm
data = positiveAssetOrIncome2.loc[:,['ASSET_AMT', 'INCOME_AMT']].as_matrix()

#separate potential donors into 10 segments based on asset amount and income amount
k_means =cluster.KMeans(n_clusters =10)

#use unsupervised knn machine learning algorithm
k_means.fit(data)
kMeansLabels = k_means.labels_
kMeansClusterCenters = k_means.cluster_centers_
kMeansLabelsUnique = np.unique(kMeansLabels)
print(kMeansLabels)
print(kMeansClusterCenters)
print(kMeansLabelsUnique)

#assign each sample to a cluster for plotting
idx, _ = vq (positiveAssetOrIncome2.loc[:,['ASSET_AMT','INCOME_AMT']], kMeansClusterCenters)

#visualize the segmentation
fig3 = plt.figure()
dotSize = 6
plt.plot(data[idx ==0,0], data[idx==0,1], 'ob', markersize = dotSize)
plt.plot(data[idx==1,0], data [idx==1,1], 'og', markersize = dotSize)
plt.plot(data[idx==2,0], data [idx==2,1], 'or', markersize = dotSize)
plt.plot(data[idx==3,0], data [idx==3,1], 'oc', markersize = dotSize)
plt.plot(data[idx==4,0], data [idx==4,1], 'om', markersize = dotSize)
plt.plot(data[idx==5,0], data [idx==5,1], 'xb', markersize = dotSize)
plt.plot(data[idx==6,0], data [idx==6,1], 'xg', markersize = dotSize)
plt.plot(data[idx==7,0], data [idx==7,1], 'xr', markersize = dotSize)
plt.plot(data[idx==8,0], data [idx==8,1], 'xc', markersize = dotSize)
plt.plot(data[idx==9,0], data [idx==9,1], 'xm', markersize = dotSize)
plt.plot(kMeansClusterCenters[:,0], kMeansClusterCenters[:,1], 'sy', markersize =10)
plt.title('Potential Donors Segmentation By Asset Amount and Income Amount Based On KNN Algorithm')
plt.xlabel('Asset Amount ($)')
plt.ylabel('Income Amount ($)')
plt.show()

#visualize median asset amount and median income amount for all 50 states
#ensure missing states records are filter out
stateData= positiveAssetOrIncome2[positiveAssetOrIncome2['STATE']!=0]
statePT = pd.pivot_table(stateData, index =['STATE'], values=['ASSET_AMT', 'INCOME_AMT'],
                         aggfunc={'ASSET_AMT': 'median', 'INCOME_AMT': 'median'})
#visualize this data in boxplot
tempData = [statePT['ASSET_AMT'].values.tolist(), statePT['INCOME_AMT'].values.tolist()]
print(tempData)
fig4 = plt.figure()
plt.boxplot(tempData)
plt.xticks([1,2],['MEDIAN ASSET AMOUNT', 'MEDIAN INCOME AMOUNT'])
plt.ylabel("value in ($)")
plt.title('2 Boxplots of Median Asset Amount and Median Income Amount for All 50 USA States')
plt.show()