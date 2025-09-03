# Les journées scientifiques de la FHU TARGET


<img src="https://raw.githubusercontent.com/rbizoi/FHU_TARGET_ScientificDays_2025/refs/heads/main/images/fhu_2025.png" width="256">

# Installation

## 01 <b></b><a href="https://www.anaconda.com/download/success">Installation Anaconda</a></b>

## 02 Installation des outils Git (optional)
### <a href="https://github.com/git-for-windows/git/releases/tag/v2.51.0.windows.1">Installation sur Windows</a>
### Installation sur Linux
```
sudo dnf install git-all
# ou 
sudo apt install git-all
```

## 02 Charger le référentiel <b>Keras</b>
<a href="https://github.com/keras-team/keras/tree/master">
     <img src="https://raw.githubusercontent.com/rbizoi/FHU_TARGET_ScientificDays_2025/refs/heads/main/images/keras_master.png" width="512">
</a>

## 03 Décompresser le fichier <b>keras_master.zip</b> dans un répertoire de travail
exemple :
```
C:\dev\keras-master
ou 
/mnt/c/dev/keras-master    
/home/utilisateur/keras-master 
```

## 04 Mise à jour des librairies de l’environnement <b>base</b>

```
conda activate root
conda update --all
python -m pip install --upgrade pip
```

## 05 Création de l’environnement <b>keras</b>

## 05.1 <b>Windows</b>

```
conda create -n keras -c conda-forge  python==3.12 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn plotly imgaug tifffile imagecodecs
```

## 05.2 <b>Linux</b>

```
conda create -p /home/razvan/anaconda3/envs/keras -c conda-forge  python==3.12 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn plotly imgaug tifffile imagecodecs
```

## 06 Configuration de l’environnement <b>keras</b>

dans le répertoire ou vous avez décompressé le fichier <b>keras_master.zip</b> exécutez les commandes suivantes 

```
conda activate keras

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
