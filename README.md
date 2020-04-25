# DED-Net-Defocus-Estimation-and-Deblurring
Code for our paper DID-ANet: Defocus Image Deblurring Network with Defocus Map Estimation as Auxiliary Task

Overview
----
Different from the object motion blur, the defocus blur is usually caused by the limitation of the cameras' depth of field. The defocus amount can be characterized by the parameter of point spread function and thus forms a defocus map. In this paper, we proposed a new network architecture called Defocus Image Deblurring Auxiliary Learning Net (DID-ANet), which is specifically designed for single image defocus deblurring by using defocus map estimation as auxiliary task to improve the deblurring result. To facilitate the training of the network, we build a new and large-scale dataset for single image defocus deblurring, which contains the defocus images, the defocus maps and the all-sharp images. To our best knowledge, DID-ANet is the first deep architecture designed for defocus image deblurring and the new dataset is the first large-scale defocus deblurring dataset for training deep networks. 

DID-ANet Architecture
----

DED Real Scene Dataset
----
Available here: https://drive.google.com/open?id=17FiFdbM6VNDBHDE4TBw7PKF7ep_wPgX6

To facilitate the training, we build the first large-scale realistic dataset for defocus map estimation and defocus image deblurring (termed DED dataset). Previously datasets is far from enough for training deep neural networks. To fill this gap, we build the dataset with the Lytro light field camera.

Training
----
run main.py

Notice: We run the training process on 4 Nvidia 1080Ti GPUs, if you want to use less GPU, the batch size should be changed.

Testing
----
run run_inference.py

Pre-trained models
----
Available here: https://drive.google.com/open?id=17FiFdbM6VNDBHDE4TBw7PKF7ep_wPgX6

Results
----

Citation
----
