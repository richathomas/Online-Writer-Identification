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
from sklearn import preprocessing


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



if __name__ == '__main__':

    Dataset = 'Dataset'
    oLabels = []
    Strokess = []
    for writer in os.listdir(Dataset):
        dataStrokes = []
        for script in os.listdir(Dataset+'/'+writer):
            fileName = Dataset+'/'+writer+'/'+script
            strokes = dataRead(fileName)
            dataStrokes.append(strokes)
        strokePoints = [item for sublist in dataStrokes for item in sublist]
        oLabels.append([writer]*len(strokePoints))
        Strokess.append(strokePoints)
    selectedStrokes = []
    selectedLabels = []
    count = 0
    for item in Strokess:
        temp = np.asarray(item)
        lTemp = np.array([oLabels[count]]).transpose()
        count+=1
        nRows = np.size(temp,0)
        remVal = nRows%30
        selectedStrokes.append(temp[:nRows-remVal,:])
        selectedLabels.append(lTemp[:nRows-remVal,:])

    filteredStrokes = []
    for item in selectedStrokes:
        filteredStrokes.append(isolatedpointremoval(item))


    subStrokeSet = []
    pointSet = 30
    for item in filteredStrokes:
        subStrokeSet.append(createsubstroke(item[:300,:],pointSet))

# Local Feature Extraction

    localFeatures = []
    for StrokeSet in subStrokeSet:
        Features = []
        for strokes in StrokeSet:
            Features.append(exctractlocalfeatures(strokes))
        localFeatures.append(Features)


    # LocalFtrs = []
    # for ftrs in localFeatures:
    #     ftrVals = []
    #     for ftr in ftrs:
    #         values = np.unique(ftr[:, 1])
    #         BinSize = values.shape[0]
    #         values, bins = np.histogram(ftr, bins=BinSize)
    #         ftrVals.append(values)
    #     LocalFtrs.append(ftrVals)

# GLobal Features
    _DglobalFeatures = []
    _AglobalFeatures = []
    for StrokeSet in subStrokeSet:
        deltaglobalFeatures = []
        aglobalFeatures = []
        for strokes in StrokeSet:
            deltagF, agF = extractglobalfeatures(strokes)
            deltaglobalFeatures.append(deltagF)
            aglobalFeatures.append(agF)
        _DglobalFeatures.append(deltaglobalFeatures)
        _AglobalFeatures.append(aglobalFeatures)

    speed = []
    for StrokeSet in subStrokeSet:
        sp = []
        for strokes in StrokeSet:
            sp.append(strokes[:,2])
        speed.append(sp)


    K = len(localFeatures)
    strokeFeature = []
    for i in range(K):
        Wftr1 = np.concatenate( localFeatures[i], axis=0 )
        Wftr2 = np.concatenate(_DglobalFeatures[i], axis=0)
        Wftr3 = np.concatenate(_AglobalFeatures[i], axis=0)
        Wftr4 = np.concatenate(speed[i],axis=0)
        ftrs = np.concatenate((Wftr1,Wftr2,Wftr3,np.array([Wftr4]).transpose()),axis=1)
        strokeFeature.append(ftrs)

    binSize = calculateentropybinsize(strokeFeature,K)
    pickle_out = open("binSize.pickle", "wb")
    pickle.dump(binSize, pickle_out)
    pickle_out.close()
    Features = []
    for i in range(K):
        tFLFeatures = []
        tFDFeatures = []
        tFAFeatures = []
        tFSFeatures = []
        lFtr = localFeatures[i]
        dFtr = _DglobalFeatures[i]
        aFtr = _AglobalFeatures[i]
        sFtr = speed[i]
        l = len(lFtr)
        for j in range(l):
            f1 = lFtr[j]
            f2 = dFtr[j]
            f3 = aFtr[j]
            f4 = sFtr[j]
            v1, bins = np.histogram(f1, bins=binSize)
            v2, bins = np.histogram(f2, bins=binSize)
            v3, bins = np.histogram(f3, bins=binSize)
            v4, bins = np.histogram(f4, bins=binSize)
            tFLFeatures.append(np.array([v1]))
            tFDFeatures.append(np.array([v2]))
            tFAFeatures.append(np.array([v3]))
            tFSFeatures.append(np.array([v4]))
        tFLFeatures = np.concatenate(tFLFeatures,axis=0)
        tFDFeatures = np.concatenate(tFDFeatures, axis=0)
        tFAFeatures = np.concatenate(tFAFeatures, axis=0)
        tFSFeatures = np.concatenate(tFSFeatures, axis=0)
        tFLFeatures =  np.reshape(tFLFeatures, (tFLFeatures.shape[0]*tFLFeatures.shape[1], 1))
        tFDFeatures = np.reshape(tFDFeatures, (tFDFeatures.shape[0] * tFDFeatures.shape[1], 1))
        tFAFeatures = np.reshape(tFAFeatures, (tFAFeatures.shape[0] * tFAFeatures.shape[1], 1))
        tFSFeatures = np.reshape(tFSFeatures, (tFSFeatures.shape[0] * tFSFeatures.shape[1], 1))
        Features.append(np.concatenate((tFLFeatures, tFDFeatures, tFAFeatures,tFSFeatures),axis=1))

    Labels = []
    alphas = []
    count = 10
    labelCount = 65
    for i in range(len(Features)):
        l = np.size(Features[i], 0)
        label = [np.unique(selectedLabels[i])[0]]*l
        alpha = generateorthogonalmatrix(Features[i].shape[0], Features[i].shape[1], labelCount)
        labelCount+=10
        Labels.append(label)
        alphas.append(alpha)
    Features = np.concatenate(Features, axis=0)
    alpha = np.concatenate(alphas, axis=0)
    Labels = np.concatenate(Labels, axis=0)

    n_components = Features.shape[0]
    resolution = Features.shape[1]
    width = binSize

    # Dictionary Learning

    F = Features
    dico = MiniBatchDictionaryLearning(n_components=n_components,
                                       alpha=resolution,
                                       n_iter=100,
                                       transform_algorithm='omp',
                                       dict_init=F)
    dl = dico.fit(F)
    fi = dl.components_
    #alpha = generateorthogonalmatrix(F.shape[0],F.shape[1],1019)

    fi = (fi-np.min(fi))/(np.max(fi)-np.min(fi))
    #alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
    F = (F - np.min(F)) / (np.max(F) - np.min(F))

    pickle_out = open("dictionary.pickle", "wb")
    pickle.dump(fi, pickle_out)
    pickle_out.close()

    Sp = np.zeros_like(F, dtype='float')
    Sn = np.zeros_like(F, dtype='float')
    n, m = F.shape
    for i in range(n):
        for j in range(m):
            if F[i, j] >= (alpha[i,j]*fi[i,j]):
                Sp[i, j] = 1 / (1 + abs(F[i, j] - (alpha[i,j]*fi[i,j])))
            else:
                Sp[i, j] = 0

            if F[i, j] < (alpha[i,j]*fi[i,j]):
                Sn[i, j] = 1 / (1 + abs(F[i, j] - (alpha[i,j]*fi[i,j])))
            else:
                Sn[i, j] = 0


    tSump = np.sum(Sp)
    tSumn = np.sum(Sn)
    SiP = np.zeros((n, 1), dtype='float')
    SiN = np.zeros((n, 1), dtype='float')
    for i in range(n):
        SiP[i, 0] = np.sum(Sp[i, :]) / tSump
        SiN[i, 0] = np.sum(Sn[i, :]) / tSumn
    FinalFeatures = np.concatenate((SiP, SiN), axis=1)
    FinalFeatures = np.add(alpha[:, :2], FinalFeatures)
    label_encoder = preprocessing.LabelEncoder()
    savedLabel = label_encoder.fit(Labels)
    Y = label_encoder.fit_transform(Labels)
    pickle_out = open("Labels.pickle", "wb")
    pickle.dump(savedLabel, pickle_out)
    pickle_out.close()


    X_train, X_test, Y_train, Y_test = train_test_split(FinalFeatures, Y,
                                                        test_size=0.2, random_state=14,
                                                        stratify=Labels)

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, Y_train)
    pickle.dump(clf, open('SVMModel.sav', 'wb'))
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(Y_test, y_pred)
    error = metrics.mean_absolute_error(Y_test, y_pred)
    precision = metrics.precision_score(Y_test, y_pred,average='micro')
    recall = metrics.recall_score(Y_test, y_pred,average='macro')
    fscore = metrics.f1_score(Y_test, y_pred,average='weighted')
    c = metrics.confusion_matrix(Y_test, y_pred)
    tp = c[0,0]
    fp = c[0,1]
    fn = c[1,0]
    tn = c[1,1]
    sensitivity = tp/(tp+fn)
    specificity = tn / (tn + fp)
    print('\n Accuracy = ', round(score * 100, 2))
    print('\n Error = ', round(error * 100, 2))
    print('\n Precision = ', round(precision * 100, 2))
    print('\n Recall = ', round(recall * 100, 2))
    print('\n Fscore = ', round(fscore * 100, 2))
    print('\n Sensitivity = ', round(sensitivity * 100, 2))
    print('\n Specificity = ', round(specificity * 100, 2))
    objects = ('Accuracy', 'Error', 'Precision', 'Recall', 'Fscore', 'Sensitivity', 'Specificity')
    y_pos = np.arange(len(objects))
    performance = [round(score * 100, 2), round(error * 100, 2),  round(precision * 100, 2), round(recall * 100, 2), round(fscore * 100, 2), round(sensitivity * 100, 2),round(specificity * 100, 2)]

    plot.bar(y_pos, performance, align='center', alpha=0.5,color=['MediumTurquoise', 'Brown', 'DarkCyan', 'DarkMagenta', 'GreenYellow','NavajoWhite','Tomato'])
    plot.xticks(y_pos, objects)
    plot.ylabel('Percentage')
    plot.title('Performance Evaluation')
    plot.show()








