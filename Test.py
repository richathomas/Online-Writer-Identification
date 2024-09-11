import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plot
import cv2 as cv
import math
import numpy as np
from sklearn.cluster import KMeans
import math
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import pandas as pd
from sklearn.decomposition import MiniBatchDictionaryLearning
import pickle
from collections import Counter

def getClass(classes):
    occurence_count = Counter(classes)
    return occurence_count.most_common(1)[0][0]

def generateorthogonalmatrix(n,m,key):
    np.random.seed(key)
    H = np.random.randn(n, m)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    mat = u @ vh
    mat = np.add(np.abs(mat),np.random.randint(low=key, high = key+20, size=(n,m)))
    return mat

def dataRead(fileName):
    handwrittenDoc = ET.parse(fileName)
    root = handwrittenDoc.getroot()
    strokeSet = root[3]
    dataSet = []
    data = []
    t1 = 0.0
    x1 = 0
    x2  = 0
    y1 = 0
    y2 = 0
    for strokes in strokeSet:
        for points in strokes:
            cords = points.attrib
            t2 = float(cords['time'])
            ts = t2 - t1
            x2 = float(cords['x'])
            y2 = float(cords['y'])
            deltaP = ((x2 - x1)**2 + (y2-y1)**2)**(1/2)
            if ts == 0:
                ts = 0.01
            speed = deltaP/ts
            if t1 != 0 and ts > 0.2:
                dataSet.append(data)
                data = []
            t1 = t2
            x1 = x2
            y1 = y2
            val = [float(cords['x']), float(cords['y']),speed]
            data.append(val)
        dataSet.append(data)
    dataSet = [item for sublist in dataSet for item in sublist]
    #strokes = np.asarray(data)
    return dataSet


def isolatedpointremoval(pointset):

    len = pointset.shape[0]
    distanceMetric = np.zeros((len, 5))

    distanceMetric[0, 0] = float('inf')
    distanceMetric[0, 1] = math.sqrt((pointset[1, 0] - pointset[0, 0]) ** 2 + (pointset[1, 1] - pointset[0, 1]) ** 2)
    distanceMetric[0, 2] = np.mean(distanceMetric[0, 1])
    distanceMetric[0, 3] = np.std(distanceMetric[0, 1])

    distanceMetric[len - 1, 0] = math.sqrt(
        (pointset[len - 1, 0] - pointset[len - 2, 0]) ** 2 + (pointset[len - 1, 1] - pointset[len - 2, 1]) ** 2)
    distanceMetric[len - 1, 1] = float('inf')
    distanceMetric[len - 1, 2] = np.mean(distanceMetric[len - 1, 0])
    distanceMetric[len - 1, 3] = np.std(distanceMetric[len - 1, 0])

    for i in range(1, len - 1):
        distanceMetric[i, 0] = math.sqrt(
            (pointset[i, 0] - pointset[i - 1, 0]) ** 2 + (pointset[i, 1] - pointset[i - 1, 1]) ** 2)
        distanceMetric[i, 1] = math.sqrt(
            (pointset[i, 0] - pointset[i + 1, 0]) ** 2 + (pointset[i, 1] - pointset[i + 1, 1]) ** 2)
        distanceMetric[i, 2] = np.mean([distanceMetric[i, 0], distanceMetric[i, 1]])
        distanceMetric[i, 3] = np.std([distanceMetric[i, 0], distanceMetric[i, 1]])

    for i in range(1, len - 1):

        if (distanceMetric[i, 0] > (distanceMetric[i, 2] + 3 * distanceMetric[i, 3])) and (
                distanceMetric[i, 1] > (distanceMetric[i, 2] + 3 * distanceMetric[i, 3])):
            distanceMetric[i, 4] = -1
        else:
            distanceMetric[i, 4] = 1
    distanceMetric[0, 4] = 1
    distanceMetric[len - 1, 4] = 1

    for i in range(len):
        if distanceMetric[i, 4] == -1:
            pointset[0, :] = np.nan

    filteredStrokes = pointset[~np.isnan(pointset).any(axis=1)]
    return filteredStrokes


def createsubstroke(dataSet, pointSet):
    strokeSet = []
    sz = np.size(dataSet,0)
    for i in range(0,sz,30):
        strokeSet.append(dataSet[i:i+30,:])
    return strokeSet

def findcentroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def exctractlocalfeatures(arr):
    cenX,cenY = findcentroid(arr)
    ftr = np.zeros((len(arr),2))
    for i in range(len(arr)):
        ftr[i,0] = math.atan((arr[i,1]-cenY)/(arr[i,0]-cenX))
        ftr[i,1] = math.sqrt(((arr[i,0]-cenX)**2)+((arr[i,1]-cenY))**2)
    return ftr


def extractglobalfeatures(arr):
    ftrrs = np.pad(arr, ((2, 2), (0, 0)), 'constant')
    ftr1 = []
    for i in range(2,len(ftrrs)-2):
        ftr1.append(sum(ftrrs[i - 2:i + 2])/5)

    ftrrs2 = np.pad(ftr1, ((2, 2), (0, 0)), 'constant')
    ftr2 = []
    for i in range(2, len(ftrrs2) - 2):
        ftr2.append(sum(ftrrs2[i - 2:i + 2]) / 5)

    ftr3 = np.zeros((len(ftr1), 2))
    for i in range(len(ftr1)):
        ftr3[i,0] = math.atan(ftr1[i][0]/ftr1[i][1])
        ftr3[i,1] = math.sqrt(ftr1[i][0]**2+ftr1[i][1]**2)

    ftr4 = np.zeros((len(ftr2), 2))
    for i in range(len(ftr2)):
        ftr4[i, 0] = math.atan(ftr2[i][0] / ftr2[i][1])
        ftr4[i, 1] = math.sqrt(ftr2[i][0] ** 2 + ftr2[i][1] ** 2)

    return ftr3, ftr4

def calculateentropybinsize(ftrs,k):
    labels = []
    for i in range(k):
        l = np.size(ftrs[i],0)
        label = np.zeros((l,1))
        label[:] = i
        labels.append(label)
    ftrs = np.concatenate(ftrs, axis=0)
    labels = np.concatenate(labels, axis=0)

    data = np.concatenate((ftrs,labels),axis=1)
    kmeans = KMeans(n_clusters=20,random_state=3425)
    kmeans.fit(ftrs)
    clusters = kmeans.fit_predict(ftrs)
    Ks = []
    for i in range(0,20):
        c = []
        for j in range(ftrs.shape[0]):
            if clusters[j] == i:
                c.append(list(data[j,:]))
        Ks.append(c)
    subcluster = []
    kmeans = KMeans(n_clusters=k,random_state=3425)
    for i in range(20):
        dt = np.array(Ks[i])
        if dt.shape[0]>=k:
            kmeans.fit(dt)
            subcluster.append(list(kmeans.fit_predict(dt)))
        else:
            cst = [1]*dt.shape[0]
            subcluster.append(cst)

    ss = []
    for i in range(0,20):
        ss1=[]
        for j in range(0,k):
            c1 = Ks[i]
            ss2 = []
            for kt in range(0,len(c1)):
                if subcluster[i][kt] == j:
                    ss2.append(list(Ks[i][kt]))
            ss1.append(ss2)
        ss.append(ss1)
    Hprob = []
    for i in ss:
        tCount = 0
        iCount = []
        for j in i:
            tCount = len(j)+tCount
            iCount.append([len(j)])
        prob = []
        for kt in iCount:
            prob.append(kt[0]/tCount)
        Hprob.append(prob)

   # Hprob = np.array(Hprob)
    Hk = []
    l = len(Hprob)
    for i in range(l):
        j = max(Hprob[i])
        if j>0:
            s= -j * math.log(j,2)
        Hk.append(s)
    fac = 1/(20*k)
    hSum = []
    for i in Hk:
        hSum.append(s*fac)

    return Hk.index(max(Hk))

def ricker_function(resolution, center, width):
    """Discrete sub-sampled Ricker (Mexican hat) wavelet"""
    x = np.linspace(0, resolution - 1, resolution)
    x = ((2 / (np.sqrt(3 * width) * np.pi ** .25))
         * (1 - (x - center) ** 2 / width ** 2)
         * np.exp(-(x - center) ** 2 / (2 * width ** 2)))
    return x

def ricker_matrix(width, resolution, n_components):
    """Dictionary of Ricker (Mexican hat) wavelets"""
    centers = np.linspace(0, resolution - 1, n_components)
    D = np.empty((n_components, resolution))
    for i, center in enumerate(centers):
        D[i] = ricker_function(resolution, center, width)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    return D

if __name__ == '__main__':

    Dataset = 'Dataset/Amper'
    strokes = dataRead('1.xml')
    Strokess = []
    strokePoints = [item for sublist in strokes for item in sublist]
    Strokess.append(strokePoints)
    selectedStrokes = []
    for item in strokes:
        selectedStrokes.append(np.asarray(item))

    nRows = np.size(selectedStrokes, 0)
    remVal = nRows % 30
    selectedStrokes = np.array(selectedStrokes)[:nRows - remVal, :]
    filteredStrokes = isolatedpointremoval(selectedStrokes)
    pointSet = 30
    subStrokeSet = createsubstroke(filteredStrokes[:300,:],pointSet)
    localFeatures = None
    for strokes in subStrokeSet:
        if localFeatures is None:
            localFeatures = exctractlocalfeatures(strokes)
        else:
            localFeatures = np.concatenate((localFeatures,exctractlocalfeatures(strokes)))
    _DglobalFeatures = None
    _AglobalFeatures = None
    for strokes in subStrokeSet:
        deltagF, agF = extractglobalfeatures(strokes)
        if _DglobalFeatures is None:
            _DglobalFeatures = deltagF
        else:
            _DglobalFeatures = np.concatenate((_DglobalFeatures,deltagF))

        if _AglobalFeatures is None:
            _AglobalFeatures = agF
        else:
            _AglobalFeatures = np.concatenate((_AglobalFeatures,agF))

    speed = []
    for StrokeSet in subStrokeSet:
        speed.append(strokes[:, 2])

    speed = np.array([np.concatenate(speed)]).transpose()

    pickle_in = open("binSize.pickle", "rb")
    binSize = pickle.load(pickle_in)

    v1, bins = np.histogram(localFeatures,bins=binSize)
    v2, bins = np.histogram(_DglobalFeatures,bins=binSize)
    v3, bins = np.histogram(_AglobalFeatures,bins=binSize)
    v4, bins = np.histogram(speed, bins=binSize)


    Features =  np.concatenate((np.array([v1]).transpose(), np.array([v2]).transpose(),np.array([v3]).transpose(),np.array([v4]).transpose()), axis=1)
    pickle_in = open("Labels.pickle", "rb")
    labels = pickle.load(pickle_in)
    alpha = []
    labelCount = 65
    for i in range(labels.classes_.shape[0]):
        alpha.append(generateorthogonalmatrix(Features.shape[0], Features.shape[1], labelCount))
        labelCount += 10

    n_components = Features.shape[0]
    resolution = Features.shape[1]
    width = binSize
    F = Features
    dico = MiniBatchDictionaryLearning(n_components=n_components,
                                       alpha=resolution,
                                       n_iter=100,
                                       transform_algorithm='omp',
                                       dict_init=F)
    dl = dico.fit(F)
    fi = dl.components_

    F = (F - np.min(F)) / (np.max(F) - np.min(F))
    loaded_model = pickle.load(open('SVMModel.sav', 'rb'))
    results = []
    scoreAvg = []
    i = 0
    for a in alpha:
        Sp = np.zeros_like(F, dtype='float')
        Sn = np.zeros_like(F, dtype='float')
        n, m = F.shape
        for i in range(n):
            for j in range(m):
                if F[i, j] >= (a[i, j] * fi[i, j]):
                    Sp[i, j] = 1 / (1 + abs(F[i, j] - (a[i, j] * fi[i, j])))
                else:
                    Sp[i, j] = 0

                if F[i, j] < (a[i, j] * fi[i, j]):
                    Sn[i, j] = 1 / (1 + abs(F[i, j] - (a[i, j] * fi[i, j])))
                else:
                    Sn[i, j] = 0


        tSump = np.sum(Sp)
        tSumn = np.sum(Sn)
        if tSump == 0:
            tSump = 0.01
        SiP = np.zeros((n, 1), dtype='float')
        SiN = np.zeros((n, 1), dtype='float')
        for i in range(n):
            SiP[i, 0] = np.sum(Sp[i, :]) / tSump
            SiN[i, 0] = np.sum(Sn[i, :]) / tSumn
        FinalFeatures = np.concatenate((SiP, SiN), axis=1)
        FinalFeatures = np.add(a[:, :2], FinalFeatures)
        result = loaded_model.predict(FinalFeatures)
        results.append(result)

        #cls = getClass(result)
    cls = np.argmax(np.sum(np.array(results),axis=1))
    user = labels.classes_[cls]
    print('Uploaded Image is of Author:-', user)




























