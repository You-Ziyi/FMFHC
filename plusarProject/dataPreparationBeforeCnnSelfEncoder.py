# -*-coding:utf-8-*-
import re
import os
import shutil
import math
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# Path: this path contains all the candidates you need to process,
# For example, I have 160000 candidates, and each folder contains 2500 candidates, so there are 64 folders in total.
# The "batchOfCandidate()" function places every 2500 features to be compressed in a folder
def batchOfCandidate(path):
    files = os.listdir(path)
    files.sort(key=natural_keys)

    for i, each in enumerate(files):
        if i % numberX == 0:
            fold_new = os.path.join(path, str(i // numberX))
            os.makedirs(fold_new)

        shutil.move(os.path.join(path, each), fold_new)

# The "createFolders()" function creates a series of folders and then places the dimensionality reduced features into these folders
def createFolders(num_chunks):

    for i in range(num_chunks):
        if i % 10 == 0:
            index = i // 10
            fold_new = "*/StorageDimensionalityReduction/" + str(index)
            os.makedirs(fold_new)

# NUM_CHUNKS = numOfCandidate / numberX ;
# numOfCandidate represents the number of all candidates to be coded
# numberX represents the number of candidates that each thread needs to code;
# How many threads are selected and how many candidates are coded by each thread are selected according to your personal computer hardware
# For example: I have 160000 candidates, and each thread encodes 2500 candidates, so I need 64 threads
numOfCandidate = 160000
numberX = 2500
NUM_CHUNKS = math.ceil(numOfCandidate/numberX)

if __name__ == '__main__':

    path = '*/FolderForStoringTVPFeatures/'
    batchOfCandidate(path)

    createFolders(NUM_CHUNKS)




