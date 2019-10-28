# DED-Net-Defocus-Estimation-and-Deblurring
Code for our paper DED-Net: the Dual-task Network for Defocus Estimation and Deblurring

Overview
----
Different  from  the  object  motion  blur  or  the  camerashake  blur,  the  situation  of  defocus  blur  varies  greatlywhen the distance between the objects and the focal planechanged, which lead to different size of defocus spread. Thelevel of defocus influence could be measure by the defocusestimation map.  In this paper,  we propose a network ap-proach to estimate the defocus level in different part of theimage, and use it to guide the deblurring task.  To train theDefocusEstimation andDebluringNetwork (DED-Net), alight-field-camera generated single image defocus deblur-ring dataset, including the blurred images, the defocus es-timations and the clear ground truth is proposed.  It’s thefirst work which pays attention to the specific situation ofblur caused by the defocus effect of cameras, and the resultsshow that our method performs the state-of-the-art resultson both the defocus estimation and the defocus deblurringtasks

DED-Net Architecture
----

DED Real Scene Dataset
----

Training
----

Testing
----

Pre-trained models
----

Results
----

Citation
----
