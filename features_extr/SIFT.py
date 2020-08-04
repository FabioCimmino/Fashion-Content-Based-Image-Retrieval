import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans as KMeans
from time import time
from sklearn.preprocessing import Normalizer
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
warnings.filterwarnings("ignore")

class SiftExtraction:
    
    @staticmethod
    # takes all images and convert them to grayscale. 
    # return a dictionary that holds all images category by category. 
    def load_images_from_folder(folder,class_pred):
        paths=list() #inizializziamo la lista dei path
        y=list() #inizializziamo la lista delle etichette
        classes= os.listdir(folder)
        classes_idx=range(len(classes))
        images = {}
        for filename in os.listdir(folder):
            category = []
            path = folder + "\\" + filename
            if (filename == class_pred):
                for cat in os.listdir(path):
                    img = cv2.imread(path + "\\" + cat,0)
                    #img = cv2.resize(img, (80,60))
                    if img is not None:
                        category.append(img)
                        y.append(classes.index(filename))
                        paths.append(path + "\\" + cat)
                images[filename] = category
        return images, paths, y
    
    @staticmethod
    # Creates descriptors using sift 
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    def sift_features(images):
        sift_vectors = {}
        descriptor_list = []
        descriptor_list_append = []
        sift = cv2.xfeatures2d.SIFT_create()
        for key,value in images.items():
            features = []
            for img in value:
               # print(key)
                kp, des = sift.detectAndCompute(img,None)
                if len(kp) < 1:
                   # print("nessun descrittore sift")
                    no_kp = np.zeros((1, sift.descriptorSize()), np.float32)
                    descriptor_list.extend(no_kp)
                    descriptor_list_append.append(no_kp)
                    features.append(no_kp)
                else:
                    #print(len(des))
                    descriptor_list.extend(des)
                    descriptor_list_append.append(des)
                    features.append(des)
            sift_vectors[key] = features
        return [descriptor_list, descriptor_list_append, sift_vectors]
    
    @staticmethod
    def create_cluster(num_centroidi, sift_descrptors,classe_predetta):
        #inizializziamo l'oggetto "KMeans" impostando il numero di centroidi
        kmeans = KMeans(num_centroidi)
        #avviamo il kmeans sulle feature estratte
        start_time=time()
        kmeans.fit(sift_descrptors)
        end_time=time()
        elapsed_time=end_time-start_time
        print ("Total time: {0:0.2f} sec.".format(elapsed_time))
        print(kmeans.cluster_centers_.shape)
        pkl_filename =os.path.dirname(__file__) + '/../models/'+"SIFT_"+classe_predetta + ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(kmeans, file)
        return kmeans
    
    @staticmethod
    def describe_dataset(descriptor_list,kmeans,num_claster):
        X=list() #inizializziamo la lista delle osservazioni
        
        for descriptor in descriptor_list:
           # print(len(descriptor))
            assignments= kmeans.predict(descriptor)
            bovw_representation, _ = np.histogram(assignments, bins=num_claster, range=(0,num_claster-1))
           # print(bovw_representation)
            X.append(bovw_representation)
        return X
       
    
    @staticmethod 
    def query_image(path,tree,idf,path_training,kmeans,num_claster):
        X=list()
        img = cv2.imread(path,0)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        if len(kp) < 1:
            # print("nessun descrittore sift")
            des = np.zeros((1, sift.descriptorSize()), np.float32)
            #descriptor_list_append.append(no_kp)
        assignments= kmeans.predict(des)
        bovw_representation, _ = np.histogram(assignments, bins=num_claster, range=(0,num_claster-1))
        X.append(bovw_representation)
        X_test_tfidf=X*idf
        norm = Normalizer(norm='l2')
        X_test_tfidf_l2 = norm.transform(X_test_tfidf)
        distance, closest_idx = tree.query(X_test_tfidf_l2,k=10)
        closest_im = []
        
         
        for dist in distance[0]:
            print(dist)
        
        for indice in closest_idx[0]:
            #print(indice)
            closest_im.append(path_training[indice])
        return closest_im
    
    @staticmethod
    def get_model (classe_predetta):
        pkl_filename =os.path.dirname(__file__) + '/../models/'+"SIFT_"+classe_predetta + ".pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model


def sift_extraction_bow (classe_predetta,img_query,dir_dataset,dirImgOut):
    images, paths, y_training= SiftExtraction.load_images_from_folder(dir_dataset,classe_predetta)
    
    sifts = SiftExtraction.sift_features(images)

    # Takes the descriptor list which is unordered one
    descriptor_list = sifts[0] 

    descriptor_list_append = sifts[1] 

    #concateno verticalmente tutti i descrittori 
    concatenated_features=np.vstack(descriptor_list)

    # Takes the sift features that is seperated class by class for train data
    all_bovw_feature = sifts[2] 

    #centroidi = SiftExtraction.create_cluster(150, concatenated_features,classe_predetta)

    centroidi = SiftExtraction.get_model(classe_predetta)
    num_claster=1250
    if (classe_predetta == 'Apparel Set' or classe_predetta == 'Cufflinks'):
        num_claster=100
    elif (classe_predetta == 'Headwear' or classe_predetta == 'Ties'):
            num_claster=150


    X = SiftExtraction.describe_dataset(descriptor_list_append,centroidi,num_claster)

    #binarizziamo il vettore di rappresentazioni X_training
    #otterremo una matrice n x 500 in cui l'elemento x_ij
    #indica se la parola j-esima era presente nell'immagine i-esima
    X=np.vstack(X)
    presence = (X>0).astype(int)
    #sommiamo tutte le righe (asse "0" della matrice)
    #ogni elemento del vettore risultate indicherà il numero di
    #righe (=immagini) in cui la parola visuale era presente
    df = presence.sum(axis=0)

    #otteniamo prima il numero di immagini
    n=len(X)
    #calcoliamo il termine secondo la formula riportata prima
    idf = np.log(float(n)/(1+df))

    X_training_tfidf=X*idf

    norm = Normalizer(norm='l2')

    X_training_tfidf_l2 = norm.transform(X_training_tfidf)

    tree = KDTree(X_training_tfidf_l2)

    closest_im = SiftExtraction.query_image(img_query,tree,idf,paths,centroidi,num_claster)

    print(closest_im)

    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    plt.axis('off')
    f, axarr = plt.subplots(2, 5)
    idx=0
    f.suptitle('SIFT Results')
    for i in range(rows):
        for j in range (columns):
            img = cv2.imread(closest_im[idx])
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axarr[i,j].imshow(img)
            axarr[i,j].axis('off')
            axarr[i,j].set_title(idx+1)
            idx+=1

    plt.savefig(dirImgOut+'\\SIFT_result.jpg')