# Les journées scientifiques de la FHU TARGET


<img src="https://raw.githubusercontent.com/rbizoi/FHU_TARGET_ScientificDays_2025/refs/heads/main/images/fhu_2025.png" width="256">

# Installation

## 01 <b></b><a href="https://www.anaconda.com/download/success">Installation Anaconda</a></b>

<b></b><a href="https://github.com/git-for-windows/git/releases/tag/v2.51.0.windows.1">GitForWindowsHelper</a></b>



<br>
<div>Mise à jour des librairies de l’environnement <b>base</b></div>

```
conda activate root
conda update --all
python -m pip install --upgrade pip
```
<div>Création de l’environnement <b>cours</b> </div>
<br>
<div><b>Windows</b> </div>
<br>

```
# conda remove -n cours --all -y
conda create -n cours -c conda-forge  python==3.10 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn plotly imgaug tifffile imagecodecs 

conda activate cours

pip install --upgrade keras tensorflow
```
<br>
<div><b>Linux</b> </div>
<br>

```
# conda remove -n cours --all -y
conda create -p /home/utilisateur/anaconda3/envs/cours -c conda-forge  python==3.10 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn plotly imgaug tifffile imagecodecs

conda activate cours

pip install --upgrade keras tensorflow
```

https://github.com/keras-team/keras/tree/master
