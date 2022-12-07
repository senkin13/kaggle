# Kaggle - Open Problems - Multimodal Single Cell Integration - 2nd Place Solution




This repository is the 2nd place solutions for the  [Kaggle - Open Problems - Multimodal Single-Cell Competition](https://www.kaggle.com/competitions/open-problems-multimodal).  

It contains  two parts from [senkin13](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/366453) and [tmp](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/366476) . 



If you run into any trouble with the setup/code or have any questions please contact [tmp](https://github.com/baosenguo) at [baosenguo@163.com](baosenguo@163.com) and [senkin13](https://github.com/senkin13) at [senkin13@hotmail.com](senkin13@hotmail.com).



---

## tmp's part

---

### OVERVIEW

This pipeline mainly consists of the following parts:

 - Preprocessing
 - FE
 - Modeling

This simple solution produced a quite robust result (Public_lb 1st; Private_lb 2nd).

### Preprocessing

- using raw count:

- normalization:

- transformation:

- standardization:

- batch-effect correction:


### Feature engineering

- decomposition

  - pca (64)
  - ipca (128)
  - factor analysis (64)

- features selection

  -  Features highly correlated with target are selected. 245 features are selected in total.

- cell-type (one-hot)


### Modelling

both mlp and lgb used the same features introduced above.

- mlp (simple mlp performs best  (single model with 1 seed - public 0.815; private 0.772))

- lgb

  

### Local CV

- random 5-fold cv
- split according to "day"

### Code

- dataset preparation 
  - /tmp/data/prepare.ipynb
  - /tmp/data/preprocess.ipynb

- training
  - /tmp/model/lgb.ipynb
  - /tmp/model/mlp.ipynb
  - /tmp/model/blending.ipynb

### requirements

  - python 3.7.5
  - pandas 1.3.5
  - numpy 1.20.3
  - torch 1.9.0
  - sklearn 1.0.2



---

## senkin13's part

---

### HARDWARE: 

(The following specs were used to create the original solution)

Windows 10 (4 TB boot disk, 64 vCPUs, 300 GB memory)
1 x NVIDIA TITAN RTX



### SOFTWARE

(python packages are detailed separately in `requirements.txt`):

Python 3.8.10
CUDA 11.3
cuddn 7.6.5.32
nvidia drivers v.466.47



### DATA SETUP

(assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)



### shell

below are the shell commands used in each step, as run from the top level directory



mkdir -p input features model sub

- download all data to input

DATA PROCESSING

- preprocess_cite.ipynb
- preprocess_multi.ipynb

TRAIN & PREDICTION

- cite_lgb_transformed_sparse_matrix.ipynb   
- cite_lgb_raw_clr_pca.ipynb
- cite_lgb_raw_sparse_matrix.ipynb
- cite_lgb_raw_target.ipynb
- multi_lgb.ipynb
- multi_nn.ipynb   

ENSEMBLE

- move tmp's tmp_cite_ensemble.joblib to ensemble/
- ensemble.ipynb

