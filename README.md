# FMFHC
1. Environmental configuration:
   
 Python 3.8+Tensorflow 2+MPI4py+PRESTO

2. Folder Introduction:
   
 (1) The codes under the "PICS" folder is used to extract the corresponding  "Time Phase Diagram" (TVP) and "Frequency Phase Diagram" (FVP) from the PFD files of the pulsar candidates.

 (2) The codes under the "DCA" folder is used for feature level fusion in multi-modal systems. By training the model, the transformation matrices of two sets of modal features are obtained.

 (3) The AX and AY in the 'plusarProject' folder are the transformation matrices of two sets of modal features obtained through DCA training. Encoder. h5 is a trained convolutional self encoding model for dimensionality reduction of TVP and FVP.

3.Operation process:

Step 1: Run the "dataPreparationBeforeCnnSelfEncoder. py" script and place all TVP or FVP feature files to be dimensionally reduced in N folders. Notice that N is number of parallel threads in Step 2 , X is the number of samples processed by each thread (users can set the value of X based on the total number of samples and the hardware device situation).

Step 2: Run the "cnnSelfEncoder. py" script to perform dimensionality reduction operations on TVP or FVP .
Step 3: Run the "dataPreparationBeforeDcaFusion. py" script, and place the compressed TVP or FVP arrays in M folders for each Y batch. Note that, M is the number of parallel threads in Step 4, and Y is the number of samples processed by each thread (users can set the value of Y according to the total number of samples and the situation of hardware devices).

Step 4: Run the "dcaFusion. py" script to fuse the statistical features of the candidates with their corresponding compressed TVP or FVP features, finally form the dataset to be tested.

Step 5: Run the "slideWindow. py" script, divide the dataset to be tested into blocks based on sliding windows, and set the window size accordingly.

Step 6: Run the 'mixedClustering.py' script to perform clustering analysis on the test dataset and obtain the final clustering result.
