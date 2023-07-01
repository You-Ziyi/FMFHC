# -*-coding:utf-8-*-
import glob
import numpy as np
import pandas as pd
import re
from keras.models import load_model
from mpi4py.futures import MPIPoolExecutor
import math
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]



# Start 64 threads, and each thread will encode and compress 2500 candidates
def myFunOfEncoder(num):

    path1 = "*/FolderForStoringTVPFeatures/" + str(num) + "/*.csv"
    files = glob.glob(path1)
    files.sort(key=natural_keys)

    context = []
    for file in files:
        context.append(pd.read_csv(file, header=None))

    data_Tvps = pd.concat(context)
    data_Tvps = np.array(data_Tvps)

    numOfCandidate = int(data_Tvps.shape[0] / 64)
    data_Tvps = data_Tvps.reshape(numOfCandidate, 64, 64, 1)

    # Loading a trained convolutional autoencoder model
    encoder = load_model("./encoder.h5")

    data_TvpsAfterEncoder = encoder.predict(data_Tvps)
    data_TvpsAfterEncoderReshape = data_TvpsAfterEncoder.reshape(data_TvpsAfterEncoder.shape[0], 64)

    index = num // 10
    path2 = "*/StorageDimensionalityReduction/" + str(index) + "/" + str(num) + ".csv"

    np.savetxt(path2, data_TvpsAfterEncoderReshape, delimiter=',') # NUM_CHUNKS = numOfCandidate / numberX ;
# numOfCandidate represents the number of all candidates to be coded
# numberX represents the number of candidates that each thread needs to code;
# How many threads are selected and how many candidates are coded by each thread are selected according to your personal computer hardware
# For example: I have 160000 candidates, and each thread encodes 2500 candidates, so I need 64 threads

allCandidates = 160000
numberX = 2500
NUM_CHUNKS = math.ceil(allCandidates/numberX)

if __name__ == '__main__':

    chunk_nums = range(NUM_CHUNKS)
    with MPIPoolExecutor() as p:
        result = p.map(myFunOfEncoder, chunk_nums)



