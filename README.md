# CDLSR
Multimodal Image Super-resolution via Joint Sparse Representations induced by Coupled Dictionaries
========================================================================

The codes correspond to the multi-stage coupled dictionary learning for multimodal image super-resolution. They are to learn a group of dictionaries or sub-dictionaries which are then used to perform guided image super-resolution for one modality, e.g. near-infrared images, with another different modality, e.g. RGB images for guidance.

The codes are freely available for research and study purposes.

Please cite:
------------
P. Song, X. Deng, J. F. Mota, N. Deligiannis, P. L. Dragotti, M. R. Rodrigues, "Multimodal Image Super-resolution via Joint Sparse Representations induced by Coupled Dictionaries", arXiv preprint arXiv:1709.08680 (2017).

P. Song, J. F. Mota, N. Deligiannis, and M. R. Rodrigues, "Coupled dictionary learning for multimodal image super-resolution", in IEEE Global Conf. Signal Inform. Process. IEEE, 2016, pp. 162–166.


Codes written & compiled by:
----------------------------
Pingfan Song
Electronic and Electrical Engineering,
University College London
uceeong@ucl.ac.uk

Packages and codes included and/or adapted:
-------------------------------------------
* OMPBox v9+ and KSVDBox v12+ by Ron Rubinstein, SPAMSv2.5 by Julien Mairal.

* Training and testing datasets are from Columbia multispectral/RGB database:
http://www.cs.columbia.edu/CAVE/databases/multispectral/
and EPFL RGB-NIR Scene database:
http://ivrl.epfl.ch/supplementary_material/cvpr11/

Usage
-----
Training:
>> Demo_MultistageCDLSR_Noise_Train.m % demo running multi-stage coupled dictionary learning (CDL) to train a group of dictionaries and sub-dictionaries.

Testing:
>> Demo_MultistageCDLSR_Test.m % demo running coupled super-resolution (CSR) to super-resolve the target image with the aid of guidance image using learning coupled dictionaries.

>> Demo_MultistageCDLSR_Noise_Test.m % demo running coupled super-resolution (CSR) to super-resolve the target image with the aid of guidance image using learning coupled dictionaries in the presence of noise.

If you want to use new images for testing and training, please replace the images in folder "Test_RGB_NIR" and "Train_RGB_NIR" by yours.


Folders
-------
Folder "Dicts" contains:
two sets of dictionaries, one for a noise-free training situation and the other for a noisy situation with the standard deviation of the training noise sigma = 12.

Folder "Results" contains:
the guided super-resolution results using the aforementioned two sets of dictionaries.

