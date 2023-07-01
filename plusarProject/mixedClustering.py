from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
import re
import time
import numpy as np
import pandas as pd
from mpi4py.futures import MPIPoolExecutor
from matplotlib import pyplot as plt

def showDeltas(rhos,deltas):
    plt.scatter(rhos[:], deltas[:], color='r', marker='+')
    plt.title("决策图")
    plt.xlabel("rhos")
    plt.ylabel("deltas")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def calculatedDensity(data, dist):

    k = 40
    nn = np.argsort(dist, axis=1)
    knn = nn[:, 1:k]
    thesize = np.size(data, 0)
    polymatri = np.zeros((thesize, k - 1))

    for i in range(thesize):

        m1 = data[i]

        for j in range(k - 1):

            m2 = data[knn[i, j]]
            # RBF_poly
            s = 1
            lamdaa = 0.95
            q = 3
            dist1 = lamdaa * np.exp(-((np.linalg.norm(m1 - m2) ** 2) / (2 * s ** 2)))
            dist2 = (1 - lamdaa) * (np.mat(m1) * np.mat(m2).T) * (
                    np.mat(m1) * np.mat(m2).T + 1) ** (q - 1)
            dist3 = dist1 + dist2
            dist = float(dist3[0][0])
            polymatri[i, j] = dist

    rho = np.sum(polymatri, axis=1)
    return rho


def calculatedDensityOfInliners(data, dist, numOfCenters):

    k = 40
    nn = np.argsort(dist, axis=1)
    knn = nn[:, 1:k]
    numOfilners = np.size(data, 0)
    polymatri = np.zeros((numOfCenters, k - 1))

    for i in range(numOfCenters):

        m1 = data[numOfilners - numOfCenters + i]

        for j in range(k - 1):
            #RBF_Poly
            m2 = data[knn[numOfilners - numOfCenters + i, j]]
            s = 1
            lamdaa = 0.95
            q = 3
            density1 = lamdaa * np.exp(-((np.linalg.norm(m1 - m2) ** 2) / (2 * s ** 2)))
            density2 = (1 - lamdaa) * (np.mat(m1) * np.mat(m2).T) * (np.mat(m1) * np.mat(m2).T + 1) ** (q - 1)
            density3 = density1 + density2
            density = float(density3[0][0])
            polymatri[i, j] = density

    rho = np.sum(polymatri, axis=1)
    return rho


def myFun(number):

    path = '*/data/' + str(number) + ".csv"
    data = pd.read_csv(path)
    pcadata = np.array(data.iloc[:, [0, 1]])

    # distset1: Mahalanobis distance between sample points
    distset1 = squareform((pdist(pcadata, 'mahalanobis')))
    distset1 = (distset1 - np.min(distset1)) / (np.max(distset1) - np.min(distset1))

    # rho is the calculated point density
    rho = calculatedDensity(pcadata, distset1)
    rhos = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))

    # delta is the closest distance with a density greater than the i-th sample point
    rankdis = distset1
    casenum = np.size(rhos)
    delta = np.zeros(casenum)
    j = np.where(rhos == max(rhos))
    for i in range(0, casenum):
        if i != j[0][0]:
            disti = rankdis[i, :]
            delta[i] = min(disti[rhos > rhos[i]])
    delta[j[0][0]] = np.max(delta)

    # Filter initial cluster center points
    a = np.where(delta > 0.025)
    b = np.where(rhos > 0.5)
    indexOfCenter = np.intersect1d(a, b)


    numOfCenter = np.size(indexOfCenter)
    centroids = pcadata[indexOfCenter]
    the_row = np.shape(pcadata)[0]
    clusterassmentt = np.mat(np.zeros((the_row, 2)))
    num_Inliners = int(the_row * 0.98)
    clusterassment = np.mat(np.zeros((num_Inliners, 2)))

    indexOfInliners = rho.argsort()[-num_Inliners:][::-1]
    maxRhoOfInliners = rho[indexOfInliners[0]]
    minRhoOfInliners = rho[indexOfInliners[-1]]
    inliners = pcadata[indexOfInliners]

    for i in range(num_Inliners):
        mindist = float('inf')
        minindex = -1
        for j in range(numOfCenter):
            oushijuli = distance.euclidean(centroids[j, :], inliners[i, :])
            dist = oushijuli
            if dist < mindist:
                mindist = dist
                minindex = j

        clusterassment[i, :] = minindex, mindist

    for cent in range(numOfCenter):

        ptsinclust = pcadata[np.nonzero(clusterassment[:, 0].A == cent)[0]]

        if len(ptsinclust) != 0:
            centroids[cent, :] = np.mean(ptsinclust, axis=0)

    # KMeans
    clusterchange = True
    sum = 0
    while sum < 10:

        sum = sum + 1
        clusterchange = False
        INliners = np.concatenate((inliners, centroids))
        distt = squareform((pdist(INliners, 'mahalanobis')))
        disttt = (distt - np.min(distt)) / (np.max(distt) - np.min(distt))
        rhoOfCenter = calculatedDensityOfInliners(INliners, disttt, numOfCenter)

        for i in range(num_Inliners):

            mindist = float('inf')
            minindex = -1

            for j in range(numOfCenter):

                oushijuli = distance.euclidean(centroids[j, :], inliners[i, :])
                gaus = np.exp(-(oushijuli ** 2) / 2 * 1 ** 2)
                rbf = np.sqrt(2 * (1 - gaus))
                dist = rbf * (((maxRhoOfInliners - minRhoOfInliners) / (maxRhoOfInliners - rhoOfCenter[j])) )

                if dist < mindist:

                    mindist = dist
                    minindex = j

            if clusterassment[i, 0] != minindex:
                clusterchange = True
                clusterassment[i, :] = minindex, mindist

        for cent in range(numOfCenter):
            ptsinclust = pcadata[np.nonzero(clusterassment[:, 0].A == cent)[0]]

            if len(ptsinclust) != 0:
                centroids[cent, :] = np.mean(ptsinclust, axis=0)

        for i in range(the_row):

            mindist = float('inf')
            minindex = -1

            for j in range(numOfCenter):

                dist = distance.euclidean(centroids[j, :], pcadata[i, :])
                if dist < mindist:

                    mindist = dist
                    minindex = j
            clusterassmentt[i, :] = minindex, mindist

    final = clusterassmentt[:, 0]

    pa1 = "D:\\plusar\\mazhi\\dddd\\test" + str(number) + ".csv"
    data_frame = pd.read_csv(path)
    data_frames = np.array(data_frame)
    out_frame = np.column_stack((data_frames, np.array(final.reshape(-1, 1))))
    pd_data = pd.DataFrame(out_frame, columns=['characteristic1', 'characteristic2', 'source', 'class', 'result'])
    pd_data.to_csv(pa1)

    data = pd.read_csv(pa1)
    groups = data.groupby('result')
    labels = [0] * len(data)
    # 按组遍历
    for index, group in groups:

        len_pulsar_all = len(group[group['class'] == 1])
        data_tmp = group[group['source'] == 66]
        len_pulsar = len(data_tmp[data_tmp['class'] == 1])
        len_all = len(group)

        if (len_pulsar / len_all) > 0.2:
            for i in group.index.tolist():
                labels[i] = 1
        else:
            for i in group.index.tolist():
                labels[i] = 0

    data['label'] = labels
    pa2 = "D:\\plusar\\mazhi\\dddd\\result\\testt" + str(number) + ".csv"
    data.to_csv(pa2, index=None)

    bank_dataset = pd.read_csv(pa2, sep=',', encoding='unicode_escape')
    y_true = np.array(bank_dataset.iloc[:, [4]]).flatten()  # lflatten()展开为一维
    y_pred = np.array(bank_dataset.iloc[:, [6]]).flatten()

    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    TP = TP.astype('float')

    # false positive
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    FP = FP.astype('float')

    # false negative
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    FN = FN.astype('float')

    # true negative
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
    TN = TN.astype('float')

    precision = (TP / (TP + FP))
    recall = (TP / (TP + FN))
    F1 = ((2 * precision * recall) / (precision + recall))
    print("accuracy：" + str((TP + TN) / (TP + TN + FP + FN)))
    print("precision：" + str((TP / (TP + FP))))
    print("recall：" + str(TP / (TP + FN)))
    print("F1：" + str((2 * precision * recall) / (precision + recall)))
    c = np.array([precision, recall, F1]).reshape(1, 3)
    with open("*/result.csv", "ab") as f:
        np.savetxt(f, c, delimiter=',')

    return final


NUM_CHUNKS = 126

if __name__ == '__main__':

    start = time.time()
    chunk_nums = range(NUM_CHUNKS)
    with MPIPoolExecutor() as p:
        result = p.map(myFun, chunk_nums)

    end = time.time()
    print("经历了:", str(end - start))



