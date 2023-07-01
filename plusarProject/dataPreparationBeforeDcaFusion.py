# -*-coding:utf-8-*-
import time
import glob
import numpy as np
import pandas as pd
import re
from mpi4py.futures import MPIPoolExecutor
import math
import os


def merge25000CandidatesIntoOneCsvFile(num):
    path = "*/StorageDimensionalityReduction/" + str(num)
    os.chdir(path)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f, low_memory=False, header=None) for f in all_filenames])
    # export to csv
    outPut_path = "*/TVP/" + str(num) + ".csv"
    combined_csv.to_csv(outPut_path, index=False, header=None)



# Each folder contains 10 csv files, and each csv file contains 2500 coded candidates.
# Merge 10 csv files into 1 csv file, so each csv file contains 25000 coded candidates.
# I put 25000 candidates in a csv file because I want to start 7
# threads to perform DCA fusion for 160000 candidates (the last thread processes 10000 candidates)
allCandidates = 160000
numberX = 2500
NUM_CHUNKS = math.ceil(allCandidates/numberX)
num = math.ceil(NUM_CHUNKS/10)

if __name__ == '__main__':

    start = time.time()

    chunk_nums = range(num)
    with MPIPoolExecutor() as p:
        result = p.map(merge25000CandidatesIntoOneCsvFile, chunk_nums)
    end = time.time()
    print("经历了:", str(end - start))


