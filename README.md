# FMFHC
1. Environmental configuration:
   
Python 3.8+Tensorflow 2+MPI4py+PRESTO

2. Folder Introduction:
   
(1) The codes under the "PICS" folder is used to extract the corresponding  "Time Phase Diagram" (TVP) and "Frequency Phase Diagram" (FVP) from the PFD files of the Pulsar candidates.

(2) The codes under the "DCA" folder is used for feature level fusion in multi-modal systems. By training the model, the transformation matrices of two sets of modal features are obtained.

(3) The AX and AY in the 'plusarProject' folder are the transformation matrices of two sets of modal features obtained through DCA training. Encoder. h5 is a trained convolutional self encoding model for dimensionality reduction of TVP and FVP.
