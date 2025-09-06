import numpy as np, pandas as pd, seaborn as sns, warnings, os, sys, pickle, time
from matplotlib import pyplot as plt
from datetime import datetime as dt

import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

import tensorflow as tf
    
__version__=0.001

palette = [
            "#030aa7", "#e50000", "#d8863b", "#005f6a", "#6b7c85", "#751973", "#d1e5f0", "#fddbc7",
            "#ffffcb", "#12e193", "#d8dcd6", "#ffdaf0", "#dfc5fe", "#f5054f", "#a0450e",
            "#0339f8", "#f4320c", "#fec615", "#017a79", "#85a3b2", "#fe2f4a", "#a00498", "#b04e0f",
            "#0165fc", "#ff724c", "#fddc5c", "#11875d", "#89a0b0", "#fe828c", "#cb00f5", "#b75203",
            "#0485d1", "#ff7855", "#fbeeac", "#0cb577", "#95a3a6", "#ffb7ce", "#c071fe", "#ca6b02",
            "#92c5de", "#f4a582", "#fef69e", "#18d17b", "#c5c9c7", "#ffcfdc", "#caa0ff", "#cb7723",
            "#d1e5f0", "#fddbc7", "#ffffcb", "#12e193", "#d8dcd6", "#ffdaf0", "#dfc5fe", "#d8863b",
            "#030764", "#be0119", "#dbb40c", "#005249", "#3c4142", "#cb0162", "#5d1451", "#653700",
            "#040348", "#67001f", "#b27a01", "#002d04", "#000000", "#a0025c", "#490648", "#3c0008"
          ]

def initParametresProjet(nomProjet,repertoireProjet):
    repertoireEnregistrement  = os.path.join(repertoireProjet,nomProjet,'model.images') 
    repertoireSauvegardes     = os.path.join(repertoireProjet,nomProjet,'model.sauvegardes') 
    
    repertoireModelCKP        = os.path.join(repertoireSauvegardes,'checkpoints')
    repertoireModelSauvegarde = os.path.join(repertoireSauvegardes,'sauvegarde')
    repertoireModelLogs       = os.path.join(repertoireSauvegardes,'tensorboard')
            
    controleExistenceRepertoire(repertoireEnregistrement)
    controleExistenceRepertoire(repertoireSauvegardes)
    
    controleExistenceRepertoire(repertoireModelCKP)
    controleExistenceRepertoire(repertoireModelSauvegarde)
    controleExistenceRepertoire(repertoireModelLogs)
    return repertoireEnregistrement, repertoireSauvegardes, repertoireModelCKP, repertoireModelSauvegarde, repertoireModelLogs

def controleExistenceRepertoire(directory, create_if_needed=True):
    """Voir si le répertoire existe. S'il n'existe pas il est créé."""
    path_exists = os.path.exists(directory)
    if path_exists:
        if not os.path.isdir(directory):
            raise Exception("Trouvé le nom "+directory+" mais c'est un fichier, pas un répertoire")
            return False
        return True
    if create_if_needed:
        os.makedirs(directory)

def sauvegarderImage( fichier, repertoireEnregistrement):
    """Enregistrez la figure. Appelez la méthode juste avant plt.show ()."""
    plt.savefig(os.path.join(repertoireEnregistrement,
                             fichier+f"--{dt.now().strftime('%Y_%m_%d_%H.%M.%S')}.png"), 
                             dpi=300, 
                             bbox_inches='tight')
    
def afficheHistoriqueEntrainement(history, palette, nom, repertoireEnregistrement=None):
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(48,16));
    markersize = 8
    linewidth=2
    
    graph = sns.lineplot(x='epoch', 
                         y='accuracy',  
                         data=history,
                         ax=ax[0],      
                         label='accuracy',
                         err_style=None, 
                         marker='o',
                         markersize=markersize,
                         linewidth=linewidth,
                         color=palette[0],
                         );
    graph = sns.lineplot(x='epoch', 
                         y='val_accuracy',  
                         data=history,
                         ax=ax[0],      
                         label='val_accuracy',
                         err_style=None, 
                         marker='o',
                         markersize=markersize,
                         linewidth=linewidth,
                         color=palette[1],
                         );
        
    ax[0].set_title(f'Accuracy {nom}', fontproperties=fm.FontProperties(size=32))
    
    graph = sns.lineplot(x='epoch', 
                         y='loss',  
                         data=history,
                         ax=ax[1],      
                         label='loss',
                         err_style=None, 
                         marker='o',
                         markersize=markersize,
                         linewidth=linewidth,
                         color=palette[0],
                         );
    graph = sns.lineplot(x='epoch', 
                         y='val_loss',  
                         data=history,
                         ax=ax[1],      
                         label='val_loss',
                         err_style=None, 
                         marker='o',
                         markersize=markersize,
                         linewidth=linewidth,
                         color=palette[1],
                         );
    ax[1].set_title(f'Loss {nom}', fontproperties=fm.FontProperties(size=32))

    if repertoireEnregistrement is not None :
        sauvegarderImage(f'afficheHistoriqueEntrainement-{nom}', repertoireEnregistrement)



def afficheMatriceConfusion(observations,predictions,dictLabels, repertoireEnregistrement, nom_essai=''):
    plt.figure(figsize=(8,8))
    sns.set(font_scale=1.5)
    sns.heatmap(pd.crosstab(observations,predictions), 
                fmt= '.0f',
                linewidths=0.3,
                #vmax=1.0, 
                square=True, 
                cmap=plt.cm.Blues,
                linecolor='white', 
                annot=True,
                cbar=False,
                xticklabels=dictLabels.values(), 
                yticklabels=dictLabels.values()
               );
    plt.xlabel('Observations', fontsize = 18);
    plt.ylabel('Prédictions', fontsize = 18);
    sauvegarderImage(f'afficheMatriceConfusion-{nom_essai}', repertoireEnregistrement)


def afficheDataset(donnees,labels,taille,image,dictLabels,cmap=None):
    plt.figure(figsize=(image, image))
    i=1
    for image, label in zip(donnees[:taille],labels[:taille]):
        ax = plt.subplot(1,taille, i)
        if cmap is None :
            plt.imshow(tf.cast(image, tf.int32))
        else :
            plt.imshow(image,cmap='gray')
        plt.title(dictLabels[int(label)])
        plt.axis("off")
        i+=1
            
def afficheProbabilites(probabilities, ind, ax, dictLabels, repertoireEnregistrement=None):
    
    prediction = pd.DataFrame(probabilities).iloc[ind,:].reset_index()
    prediction.columns = ['Classe','Probabilite']
    prediction.Classe = prediction.Classe.apply(lambda x : f'{x:02d}-{dictLabels[x]}')
    
    graph = sns.barplot(
                        x='Classe',
                        y='Probabilite',
                        data=prediction, #.sort_values('Probabilite',ascending=False),
                        palette=palette,
                        ax=ax        
                        );
    graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
    ax.set_xlabel('');
    ax.set_ylabel('');    
    
    for i, patche in enumerate(graph.patches):
        if patche.get_height() > 0 :
            graph.text(
                        patche.get_x()+0.4,
                        0.4, #2*patche.get_height()/3,
                        f'{patche.get_height()*100:0.4f}%',
                        color='black',
                        rotation='vertical',
                        # size='large',
                        fontsize='large',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                        verticalalignment='center',
                        horizontalalignment='center',
                       )       
    if repertoireEnregistrement is not None :            
        sauvegarderImage('Probabilités pour les 4 premiers prédictions', repertoireEnregistrement)

def afficheDistributionsPipe(pipeline, dictLabels, palette, repertoireEnregistrement=None):
    fig, ax = plt.subplots(figsize=(24,12));
    for i, (_,lab) in enumerate(pipeline): 
        labels = lab if i==0 else tf.concat([labels,lab], 0)
    
    affichage = pd.DataFrame(labels.numpy(),columns=['label'])
    affichage['nom'] = affichage['label'].apply(lambda x: dictLabels[x])
    affichage = affichage.groupby(['nom']).label.count().reset_index().rename(columns={'label':'nombre'})
    affichage['%'] = affichage.nombre * 100 / affichage.nombre.sum()
    
    graph = sns.barplot(x="nom",y='nombre', data=affichage, palette=palette,  ax=ax)
    loc, labels = plt.xticks()
    graph.set_xticklabels(labels, rotation=90);
    
    for patche in graph.patches:
        if patche.get_height() > 0 :
            graph.text(
                        patche.get_x() + patche.get_width() / 2 ,
                        affichage['nombre'].mean()/2, 
                        f"{int(patche.get_height())} - {patche.get_height()*100/affichage.nombre.sum():0.2f}%",
                        color='black',
                        rotation='vertical',
        #                 size='large',
        #                 fontsize='large',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                        verticalalignment='center',
                        horizontalalignment='center',
                       )  
                
    ax.set_xlabel('');
    ax.set_ylabel('');
    ax.set_title(f'Distributions des classes -total {affichage.nombre.sum()}', fontproperties=fm.FontProperties(size=32))
    if repertoireEnregistrement is not None :            
        sauvegarderImage('Distributions des classes', repertoireEnregistrement)

def afficheDistributions(donnees, dictLabels, palette, repertoireEnregistrement=None):
    fig, ax = plt.subplots(figsize=(24,12));
    
    affichage = pd.DataFrame(donnees["train_labels"].ravel(),columns=['label'])
    affichage['nom'] = affichage['label'].apply(lambda x: dictLabels[x])
    affichage = affichage.groupby(['nom']).label.count().reset_index().rename(columns={'label':'nombre'})
    affichage['%'] = affichage.nombre * 100 / affichage.nombre.sum()
    
    graph = sns.barplot(x="nom",y='nombre', data=affichage, palette=palette,  ax=ax)
    loc, labels = plt.xticks()
    graph.set_xticklabels(labels, rotation=90);
    
    for patche in graph.patches:
        if patche.get_height() > 0 :
            graph.text(
                        patche.get_x() + patche.get_width() / 2 ,
                        affichage['nombre'].mean()/2, 
                        f"{int(patche.get_height())} - {patche.get_height()*100/affichage.nombre.sum():0.2f}%",
                        color='black',
                        rotation='vertical',
        #                 size='large',
        #                 fontsize='large',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                        verticalalignment='center',
                        horizontalalignment='center',
                       )  
                
    ax.set_xlabel('');
    ax.set_ylabel('');
    if repertoireEnregistrement is not None :            
        sauvegarderImage('Distributions des classes', repertoireEnregistrement)    


if (__name__ == "__main__"):
    print(__version__)


