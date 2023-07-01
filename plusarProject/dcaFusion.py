# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
from mpi4py.futures import MPIPoolExecutor
import math
from sklearn.preprocessing import MinMaxScaler

"""
%   Details can be found in:
%   
%   M. Haghighat, M. Abdel-Mottaleb, W. Alhalabi, "Discriminant Correlation 
%   Analysis: Real-Time Feature Level Fusion for Multimodal Biometric 
%   Recognition," IEEE Transactions on Information Forensics and Security, 
%   vol. 11, no. 9, pp. 1984-1996, Sept. 2016. 
%   http://dx.doi.org/10.1109/TIFS.2016.2569061
% 
%   and
%   M. Haghighat, M. Abdel-Mottaleb W. Alhalabi, "Discriminant Correlation 
%   Analysis for Feature Level Fusion with application to multimodal 
%   biometrics," IEEE International Conference on Acoustics, Speech and 
%   Signal Processing (ICASSP), 2016, pp. 1866-1870. 
%   http://dx.doi.org/10.1109/ICASSP.2016.7472000
% 
% 
% 
% (C)   Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       PLEASE CITE THE ABOVE PAPERS IF YOU USE THIS CODE.
"""

def myFun(num):
    path1 = "*/HTRU2_statistic/" + str(num) + ".csv"
    path2 = "*/TVP/" + str(num) + ".csv"

    plusarFeature = np.array(pd.read_csv(path1, header=None))
    data_TvpEncoded = np.array(pd.read_csv(path2, header=None))

    # Ax: Transformation matrix for the first data set (rxp)
    # Ay: Transformation matrix for the second data set (rxq)
    path3 = "./Ax.csv"
    path4 = "./Ay.csv"

    Ax = np.array(pd.read_csv(path3))
    Ay = np.array(pd.read_csv(path4))

    tool = MinMaxScaler(feature_range=(0, 1))
    plusarFeature = tool.fit_transform(plusarFeature)

    result1 = np.matmul(Ax, (plusarFeature.T))
    result2 = np.matmul(Ay, (data_TvpEncoded.T))
    re = np.concatenate((result1, result2)).T

    path5 = "*/res" + str(num) + ".csv"
    np.savetxt(path5, re, delimiter=',')

NUM_CHUNKS = math.ceil(64/10)
if __name__ == '__main__':

    chunk_nums = range(NUM_CHUNKS)  # 100 chunks
    with MPIPoolExecutor() as p:
        result = p.map(myFun, chunk_nums)



