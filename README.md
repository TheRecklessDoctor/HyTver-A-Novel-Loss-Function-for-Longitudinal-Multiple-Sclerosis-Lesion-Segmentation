# HyTver
by Dayan Perera, Ting Fung Fung and Vishnu Monn Baskaran.



### Introduction
This repository is for our paper "HyTver:  A Novel Loss Function for Longitudinal Multiple Sclerosis Lesion Segmentation.", which was accepted at APSIPA 2025.

### Environment
This repository is based on PyTorch 2.4.0, CUDA 12.4, and Python 3.11.0. All experiments in our paper were conducted on a single NVIDIA A5000 GPU with the experiment settings being identical. 

### Data and Preparation
Please follow the instructions given [here](https://github.com/ycwu1997/CoactSeg/) for the data preparation and data split. The dataset itself is the [MSSEG-2](https://portal.fli-iam.irisa.fr/msseg-2/data/) which you have to obtain from the site. To perform skull-stripping please download and use [HD-BET](https://github.com/MIC-DKFZ/HD-BET). We use the python package version as specified in the repository. Detailed steps to preprocess the data are provided below.

1. Clone repository
```
git clone https://github.com/TheRecklessDoctor/HyTver 
```

2. add a folder inside CoactSeg called model and another one inside model called vnet
```
cd CoactSeg
mkdir model
cd model
mkdir vnet
```

3. Obtain original data from [MSSEG-2](https://portal.fli-iam.irisa.fr/msseg-2/data/) website. The original data should be saved into a folder alongside the CoactSeg folder.

4. Get HD-BET as per instructions given in their [repository](https://github.com/MIC-DKFZ/HD-BET). If you are cloning the repo and not installing the package, HD-BET should be cloned alongside the CoactSeg folder and not inside it.

5. Run HD-BET skull-stripping as follows  
```
cd HD-BET
bash ../run-HD-BET.sh name_of_original_data_folder name_of_folder_to_store_skull-stripped_data
```

6. Go back to the top-level directory and copy skull-stripped folder to data folder inside CoactSeg
```
cd ../
cp -r name_of_skull-stripped_folder ./CoactSeg/data/MSSEG2/h5/
```

7. Modify the file "./CoactSeg/data/MSSEG2/h5/pre_processing.py" to include the paths to the data folders as specified in the file pre_processing.py.

8. Run the preprocessing script
```
bash runPreprocess.sh
```


### Running the model

1. Go into the CoactSeg folder
```
cd CoactSeg
```

2. Run the training and testing script
```
bash train_mixed.sh
```

3. However if you want to change the loss function used then modify train_mixed.sh and change the parameter "loss-func". The options are given below:  
    * hyTver
    * crossEntropy
    * dice
    * focal
    * focalDice
    * symmetricFocal
    * asymmetricFocal
    * symmetricFocTv
    * asymmetricFocTv 
    * symmetricUnifiedFoc
    * asymmetricUnifiedFoc
    * tversky
    * focalTversky
    * logcosh
    * comboloss
    * weightedce
    * dicepp - we didn't test completely as the time to run an iteration was significantly greater compared to the others.
    * hdloss
