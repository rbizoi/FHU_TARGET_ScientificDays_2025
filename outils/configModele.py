import numpy as np, pandas as pd, seaborn as sns, warnings, os, sys, pickle, time
from matplotlib import pyplot as plt
from datetime import datetime as dt

import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

from sklearn.preprocessing import label_binarize
from sklearn.metrics import make_scorer, confusion_matrix, roc_curve, auc, accuracy_score, log_loss, hamming_loss, \
                            precision_score, recall_score, f1_score, fbeta_score, jaccard_score, \
                            precision_recall_curve, average_precision_score, balanced_accuracy_score, \
                            classification_report,roc_auc_score

import tensorflow as tf
import keras

from configProjet import sauvegarderImage

__version__=0.001

def preprocessionID(x):return x

modelDictionnaire = {
                      # 'VGG19':{'model':keras.applications.VGG19,'preprocess':keras.applications.vgg19.preprocess_input},      
                      'ResNet152V2':{'model':keras.applications.ResNet152V2,'preprocess':keras.applications.resnet_v2.preprocess_input},  
                      'DenseNet201':{'model':keras.applications.DenseNet201,'preprocess':keras.applications.densenet.preprocess_input},  
                      'Xception':{'model':keras.applications.Xception,'preprocess':keras.applications.xception.preprocess_input},  
                      'InceptionV3':{'model':keras.applications.InceptionV3,'preprocess':keras.applications.inception_v3.preprocess_input},  
                      'InceptionResNetV2':{'model':keras.applications.InceptionResNetV2,'preprocess':keras.applications.inception_resnet_v2.preprocess_input},
                      'MobileNetV3Small':{'model':keras.applications.MobileNetV3Small,'preprocess':keras.applications.mobilenet_v3.preprocess_input},
                      'EfficientNetV2S':{'model':keras.applications.EfficientNetV2S,'preprocess':preprocessionID},  
                      'ConvNeXtSmall':{'model':keras.applications.ConvNeXtSmall,'preprocess':preprocessionID},  
                    }

def initParametresExecution(image_size=(160, 160, 3),
                            batch_size = 32,
                            epochs=1024,
                            repertoire='../donnees/WhiteBloodCells-8-tfrecords',
                            cycle_length=None,
                            deterministic=None,
                            repetition=1,
                            bufferAleatoire=1024):

    # with open(os.path.join(repertoire,'dictLabels'), 'wb') as fichier:
    #     pickle.dump(dictLabels, fichier)
    with open(os.path.join(repertoire,'dictLabels'), 'rb') as fichier:
        dictLabels = pickle.load(fichier)
    
    nombreClasses=len(dictLabels.keys())
    pipeline_apprentissage = getPipelineDataset(repertoire= os.path.join(repertoire,'apprentissage','*'),
                                                image_size=image_size,
                                                batch_size=batch_size,
                                                apprentissage=True,
                                                cycle_length=None,
                                                deterministic=None,
                                                repetition=repetition,
                                                bufferAleatoire=1024)
    pipeline_validation = getPipelineDataset(repertoire= os.path.join(repertoire,'validation','*'),
                                                image_size=image_size,
                                                batch_size=batch_size,
                                                apprentissage=True,
                                                cycle_length=None,
                                                deterministic=None,
                                                repetition=1,
                                                bufferAleatoire=1024)    
    
    return image_size,batch_size,epochs,dictLabels,nombreClasses,pipeline_apprentissage,pipeline_validation


def getPipelineDataset(repertoire,
                       image_size,
                       batch_size,
                       apprentissage=True,
                       cycle_length=None,
                       deterministic=None,
                       repetition=1,
                       bufferAleatoire=1024):
    def _traitementImage(image):
        image = tf.image.decode_jpeg(image, channels=image_size[2])
        return image
    
    def _lectureTFRecord(enregistrement):
        formatTFRecord = ({
            'image': tf.io.FixedLenFeature([], tf.string),
            'label':  tf.io.FixedLenFeature([], tf.int64),
        })
    
        enregistrement = tf.io.parse_single_example(enregistrement, formatTFRecord)
        enregistrement['image'] =  _traitementImage(enregistrement['image'])    
        return enregistrement
    
    def _dictToImageLabel(enregistrement):
        return enregistrement['image'],enregistrement['label']
        
    pipeline = tf.data.Dataset.list_files(repertoire,shuffle=True)
    pipeline = pipeline.interleave( tf.data.TFRecordDataset,
                                    cycle_length=cycle_length,
                                    num_parallel_calls=tf.data.AUTOTUNE, 
                                    deterministic=deterministic)
    if apprentissage : pipeline = pipeline.repeat(repetition)
    pipeline = pipeline.shuffle(1024)
    pipeline = pipeline.map(_lectureTFRecord)
    pipeline = pipeline.map(_dictToImageLabel)
    pipeline = pipeline.map(lambda image, label: (tf.image.resize(image, image_size[:2]), label))
    pipeline = pipeline.batch(batch_size)
    pipeline = pipeline.prefetch(tf.data.AUTOTUNE)                  

    return pipeline

# def evolutionTauxApprentissage(epoch,
#                                initial_lrate=1e-03,
#                                max_lrate=1e-03,
#                                min_lrate=1e-05,
#                                epochs_runup_lrate=0,
#                                epochs_sustain_lrate=0,
#                                drop_lrate=0.96):
#     # epochs_drop = epochs * 0.1
#     if epoch < epochs_runup_lrate:
#         lrate = (max_lrate - initial_lrate) / epochs_runup_lrate * epoch + initial_lrate
#     elif epoch < epochs_runup_lrate + epochs_sustain_lrate:
#         lrate = max_lrate
#     else:
#         lrate = (max_lrate - min_lrate) * drop_lrate**(epoch - epochs_runup_lrate - epochs_sustain_lrate) + min_lrate
#     return lrate


def creationCompilationModele(nomModel, 
                              modelDictionnaire, 
                              image_size, 
                              nombreClasses, 
                              optimizer=keras.optimizers.Adam(1e-03)
                             ):
    pretrained_model = modelDictionnaire[nomModel]['model'](weights=None, include_top=False)
    
    inputs = keras.Input(shape=image_size, name='input_layer')
    x = modelDictionnaire[nomModel]['preprocess'](inputs)    
    x = pretrained_model(x)    
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(nombreClasses, activation='softmax', name='couche_prob')(x)
    
    model = keras.Model(inputs, outputs, name=f'{nomModel}_blood')

    model.compile(
        optimizer=optimizer, 
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )
    
    return model

def creationRappelsExecution( repertoireModelSauvegarde, 
                              repertoireModelCKP,
                              repertoireModelLogs, 
                              patienceArretPrecoce=None,
                              initial_lrate=1e-03,
                              max_lrate=1e-03,
                              min_lrate=1e-05,
                              epochs_runup_lrate=0,
                              epochs_sustain_lrate=0,
                              drop_lrate=0.96
                              ):
    
    def evolutionTauxApprentissage(epoch,
                                   initial_lrate=initial_lrate,
                                   max_lrate=max_lrate,
                                   min_lrate=min_lrate,
                                   epochs_runup_lrate=epochs_runup_lrate,
                                   epochs_sustain_lrate=epochs_sustain_lrate,
                                   drop_lrate=drop_lrate):
        # epochs_drop = epochs * 0.1
        if epoch < epochs_runup_lrate:
            lrate = (max_lrate - initial_lrate) / epochs_runup_lrate * epoch + initial_lrate
        elif epoch < epochs_runup_lrate + epochs_sustain_lrate:
            lrate = max_lrate
        else:
            lrate = (max_lrate - min_lrate) * drop_lrate**(epoch - epochs_runup_lrate - epochs_sustain_lrate) + min_lrate
        return lrate
    
    filename = os.path.join(repertoireModelSauvegarde, 'modelVLoss.keras')
    learningRate = keras.callbacks.LearningRateScheduler(evolutionTauxApprentissage)
    checkpointLoss = keras.callbacks.ModelCheckpoint(filename, 
                                                        monitor = 'val_loss',
                                                        verbose = 1, 
                                                        save_best_only = True, 
                                                        mode = 'min')
    
    backupAndRestore = keras.callbacks.BackupAndRestore(backup_dir=repertoireModelCKP, delete_checkpoint=False)
    tensorBoard = keras.callbacks.TensorBoard(log_dir=repertoireModelLogs)

    if patienceArretPrecoce is not None :        
        return [learningRate,checkpointLoss,backupAndRestore,tensorBoard,
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienceArretPrecoce, verbose=1)]
    else :
        return [learningRate,checkpointLoss,backupAndRestore,tensorBoard]
        
    

def entrainementModele( model, 
                        pipelineApprentissage, 
                        pipelineValidation, 
                        epochs, 
                        batch_size,
                        callbacks,
                        verbose, 
                        repertoireModelSauvegarde):
    t1 = time.time()  
    
    model_history = model.fit(pipelineApprentissage, 
                        validation_data=pipelineValidation, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=verbose
                       )
    
    model = tf.keras.models.load_model(os.path.join(repertoireModelSauvegarde,'modelVLoss.keras'))
    print(f"Exécution  : {(time.time() - t1)/60:0.2f} m")
    return model_history, model

def sauvegarderModel(model, fichier, repertoireSauvegardes):
    """Enregistrez le modèle Keras."""
    if fichier != None:
        controleExistenceRepertoire(repertoireSauvegardes)
        nomFichier = os.path.join(repertoireSauvegardes, '{}.keras'.format(fichier))
        model.save(nomFichier)    

def sauvegardeHistorique(model,
                         repertoireSauvegardes,
                         nomSauvegarde='one_hidden_layer_history_batch_size_1'):

    history = pd.DataFrame( model.history)
    history.reset_index(inplace=True)
    history.rename(columns={'index':'epoch'},inplace=True)
    history.epoch += 1
    history.to_parquet(os.path.join(repertoireSauvegardes,f'{nomSauvegarde}.gzip'),compression='gzip', engine='pyarrow') 
    return history


def executeApprentissageChoixClassifieurs(model,
                                          X_test,
                                          y_test,
                                          label_dict,
                                          palette,
                                          repertoireEnregistrement,
                                          nom_essai=''
                                         ):
    
    def afficheCourbes(vraisPositifs,fauxPositifs,aucROCt,precisions,sensibilites,avgPrecRec,nbClasses,lw,label_dict):
        plt.figure(figsize=(24, 24));
        for i, color in zip(range(nbClasses), palette):
            plt.plot(fauxPositifs[i], vraisPositifs[i], color=color, lw=lw,
                     label=' ' + label_dict[i] + ' (AUC = {1:0.8f})'
                                                             ''.format(i, aucROCt[i]))
    
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Taux de faux Positifs-(1 - Spécificité) = VN / (FP + VN)',size=18)
        plt.ylabel('Taux de vrais positifs-Sensibilité = VP / (VP + FN)',size=18)
        plt.title('Courbe ROC (Receiver Operating Caracteristic)',size=20)
        plt.legend(loc="lower right"); #, fontsize='large'
        sauvegarderImage(f'Courbe ROC-{nom_essai}', repertoireEnregistrement)
        
        plt.figure(figsize=(24,24));
    
        f_scores = np.linspace(0.2, 0.9, num=8)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
        for i, color in zip(range(nbClasses), palette):
            plt.step(sensibilites[i], 
                         precisions[i], 
                         where='post', 
                         color=color, 
                         lw=lw, 
                         label=f"{label_dict[i]}(APR = {avgPrecRec[label_dict[i]]:0.8f})"
                    )
            plt.fill_between(sensibilites[i], precisions[i], step='post', alpha=0.05)            
    
            # plt.plot(fauxPositifs[i], vraisPositifs[i], color=color, lw=lw,
            #          label=' ' + label_dict[i] + ' (AUC = {1:0.8f})'
            #                                                  ''.format(i, aucROCt[i]))        
    
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])        
        plt.xlabel('Sensibilité/Rappel(Recall) = VP / (VP + FN)',size=18)
        plt.ylabel('Précision = VP / (VP + FP)',size=18)        
        plt.title('Courbe Précision-Rappel',size=20)
        plt.legend(loc="lower right") # , fontsize = 'large'
        sauvegarderImage(f'Courbe Précision-Rappel-{nom_essai}', repertoireEnregistrement)
    
    cvF1, cvF1SD, cvAccuracy, cvAccSD, aucROC, avgPrecRec, accuracy, balanced_accuracy, logloss, hammingloss, precision, sensibilite, \
    f1, f2, f05, jaccard, vrais_negatifs, faux_positifs, faux_negatifs, vrais_positifs, total_positifs, aucROCtn = \
        dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(),\
        dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
    #
    oneloss, precision_micro, precision_macro, precision_weighted, \
    sensibilite_macro, sensibilite_micro, sensibilite_weighted, \
    f1_micro, f1_macro, f1_weighted,f2_micro,f2_macro,f2_weighted,f05_micro,f05_macro,f05_weighted = \
        dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), \
        dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
    
    fauxPositifs, vraisPositifs, precisions, sensibilites, aucROCt, pr_auc, tauxROC, tauxPR = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()        
    
    
    lw = 1
    nbClasses   = len(label_dict.keys())
    listClasses = list(label_dict.keys())
    
    y_testA  = label_binarize(y_test, classes=listClasses)
    plt.figure(figsize=(18,18))
    
    
    t1 = time.time()  
    classifier = model
    
    # y_score     = model.predict_proba(X_test)
    # y_pred      = model.predict(X_test)
    y_score     = model.predict(X_test)
    y_pred      = np.argmax(y_score, axis=-1) 
    y_predA     = label_binarize(y_pred, classes=listClasses)
    
    accuracy['global']              = accuracy_score(y_test, y_pred)
    balanced_accuracy['global']     = balanced_accuracy_score(y_test, y_pred)
    precision['global']             = precision_score(y_test, y_pred, average='weighted')
    sensibilite['global']           = recall_score(y_test, y_pred, average='weighted')
    
    f1['global']                    = f1_score(y_test, y_pred, average='weighted')
    f2['global']                    = fbeta_score(y_test, y_pred, beta=2, average='weighted')
    f05['global']                   = fbeta_score(y_test, y_pred, beta=0.5, average='weighted')
    
    vrais_negatifs['global']        = 0
    faux_positifs ['global']        = 0
    faux_negatifs ['global']        = 0
    vrais_positifs['global']        = 0
    total_positifs['global']        = 0
    
    
    aucROC['global'] = roc_auc_score(y_test, y_score, multi_class='ovr')
    
    for i in range(nbClasses):
        fauxPositifs[i], vraisPositifs[i], tauxROC[i] = roc_curve(y_testA[:, i], y_score[:, i])
        aucROCt[i]                                    = auc(fauxPositifs[i], vraisPositifs[i])
        precisions[i], sensibilites[i], tauxPR[i]     = precision_recall_curve(y_testA[:, i], y_score[:, i])
    
        aucROC[label_dict[i]]                = aucROCt[i]
        avgPrecRec[label_dict[i]]            = average_precision_score(y_testA[:, i], y_score[:, i])
        accuracy[label_dict[i]]              = accuracy_score(y_testA[:, i], y_predA[:, i])
        balanced_accuracy[label_dict[i]]     = balanced_accuracy_score(y_testA[:, i],y_predA[:, i])
        logloss[label_dict[i]]               = log_loss(y_testA[:, i], y_predA[:, i])
        hammingloss[label_dict[i]]           = hamming_loss(y_testA[:, i], y_predA[:, i])
        precision[label_dict[i]]             = precision_score(y_testA[:, i], y_predA[:, i])
        sensibilite[label_dict[i]]           = recall_score(y_testA[:, i], y_predA[:, i])
        f1[label_dict[i]]                    = f1_score(y_testA[:, i], y_predA[:, i])
        f2[label_dict[i]]                    = fbeta_score(y_testA[:, i], y_predA[:, i], beta=2)
        f05[label_dict[i]]                   = fbeta_score(y_testA[:, i], y_predA[:, i], beta=0.5)
      
        jaccard[label_dict[i]]               = jaccard_score(y_testA[:, i], y_predA[:, i])
        vrais_negatifs[label_dict[i]]        = confusion_matrix(y_testA[:, i], y_predA[:, i])[0, 0]
        faux_positifs[label_dict[i]]         = confusion_matrix(y_testA[:, i], y_predA[:, i])[0, 1]
        faux_negatifs[label_dict[i]]         = confusion_matrix(y_testA[:, i], y_predA[:, i])[1, 0]
        vrais_positifs[label_dict[i]]        = confusion_matrix(y_testA[:, i], y_predA[:, i])[1, 1]
        total_positifs[label_dict[i]]        = vrais_positifs[label_dict[i]] + faux_negatifs [label_dict[i]]
        vrais_negatifs['global']              += vrais_negatifs[label_dict[i]]
        faux_positifs ['global']              += faux_positifs [label_dict[i]]
        faux_negatifs ['global']              += faux_negatifs [label_dict[i]]
        vrais_positifs['global']              += vrais_positifs[label_dict[i]]
    
    
    total_positifs['global'] = vrais_positifs['global'] + faux_negatifs ['global']
    
    
    fauxPositifs["micro"], vraisPositifs["micro"], _ = roc_curve(y_testA.ravel(), y_score.ravel())
    aucROCt["micro"]                                 = auc(fauxPositifs["micro"], vraisPositifs["micro"])
    
    listFauxPositifs = np.unique(np.concatenate([fauxPositifs[i] for i in range(nbClasses)]))
    moyenneVraisPositifs = np.zeros_like(listFauxPositifs)
    for i in range(nbClasses):
        moyenneVraisPositifs += np.interp(listFauxPositifs, fauxPositifs[i], vraisPositifs[i])
    
    moyenneVraisPositifs /= nbClasses
    
    fauxPositifs["macro"], vraisPositifs["macro"] = listFauxPositifs, moyenneVraisPositifs
    aucROCt["macro"] = auc(fauxPositifs["macro"], vraisPositifs["macro"])
    # aucROC['global'] = aucROCt["macro"]  # (aucROCt["micro"],aucROCt["macro"])
    
    avgPrecRec['global'] = average_precision_score(y_testA.ravel(), y_score.ravel(), average='weighted')
    
    afficheCourbes(vraisPositifs,fauxPositifs,aucROCt,precisions,sensibilites,avgPrecRec,nbClasses,lw,label_dict);
    
    print ("Area under the ROC curve : %0.4f" % aucROC['global'],end='\t--\t')
    print('Exécution  :'+('%.2fs' % (time.time() - t1)).lstrip('0'))
        
    resultats = pd.DataFrame(pd.Series(aucROC), columns=["aucROC"])
    resultats["avgPrecRec"]              = pd.Series(avgPrecRec)
    resultats["f1"]                      = pd.Series(f1)
    resultats["f2"]                      = pd.Series(f2)
    resultats["f05"]                     = pd.Series(f05)
    resultats["accuracy"]                = pd.Series(accuracy)
    
    resultats["precision"]               = pd.Series(precision)
    resultats["sensibilite"]             = pd.Series(sensibilite)
    resultats["vrais_positifs"]          = pd.Series(vrais_positifs)
    resultats["vrais_negatifs"]          = pd.Series(vrais_negatifs)
    resultats["faux_positifs"]           = pd.Series(faux_positifs)
    resultats["faux_negatifs"]           = pd.Series(faux_negatifs)
    resultats["total_positifs"]          = pd.Series(total_positifs)
    
    resultats.reset_index(inplace=True)
    resultats.rename(columns={"index": "Classe"}, inplace=True)
    resultats['essai'] = nom_essai
    return resultats

def modelPersonalise(image_size, nombreClasses, listeFiltres=[128,128,256,512], dropout=0.5, activation=keras.ops.gelu):
    preprocessData = tf.keras.Sequential([
        keras.layers.Rescaling(1.0 / 255),
    ])
    
    # Squeezenet architecture
    def blockSqueezeNet(x, squeeze, expand):
        
        y  = keras.layers.Conv2D(filters=squeeze, kernel_size=1, padding='same')(x)
        y  = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation(activation)(y)
        
        y1 = keras.layers.Conv2D(filters=expand//2, kernel_size=1, padding='same')(y)
        y1 = keras.layers.BatchNormalization()(y1)
        y1 = keras.layers.Activation(activation)(y1)
        
        y3 = keras.layers.Conv2D(filters=expand//2, kernel_size=3, padding='same')(y)
        y3 = keras.layers.BatchNormalization()(y3)
        y3 = keras.layers.Activation(activation)(y3)
    
        z = keras.layers.SeparableConv2D(expand//2, 3, padding="same")(x)
        z = keras.layers.BatchNormalization()(z)
        z = keras.layers.Activation(activation)(z)
        
        return keras.layers.concatenate([y1, y3, z])
            

    inputs = keras.Input(shape=image_size)
    
    x = preprocessData(inputs)
    
    x = keras.layers.Conv2D(listeFiltres[0], 3, strides=1, padding="same",kernel_initializer='glorot_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    previous_block_activation = x  # Set aside residual
    
    for size in listeFiltres[1:]:
        x = blockSqueezeNet(x, size, size*2)    
        x = keras.layers.Conv2D(size, 3, strides=2, padding="same")(x)
        
        residual = keras.layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    
    x = keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(nombreClasses, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)


if (__name__ == "__main__"):
    print(__version__)


