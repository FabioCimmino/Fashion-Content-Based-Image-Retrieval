from keras.models import load_model
from keras.models import Model
import numpy as np
from keras.preprocessing import image
import os
import cv2
import matplotlib.pyplot as plt
import pickle

def deep_method (img_class, img_query, dir_dataset,dirImgOut):

    img = image.load_img(img_query, target_size=(80, 60))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    model = load_model(os.path.dirname(__file__) +'/../models/modello.h5')
    # Get encoder layer from trained model

    encoder = Model(inputs=model.input, outputs=model.layers[3].output)

    def euclidean(a, b):
	    # compute and return the euclidean distance between two vectors
	    return np.linalg.norm(a - b)
    
    features_query=encoder.predict(img_tensor)

    dire= dir_dataset+'\\'+ img_class

    features_dataset= []
    """
    for img in os.listdir(dire):
        dataset_dir = os.path.join(dire, img)
        img_dataset = image.load_img(dataset_dir, target_size=(80, 60))
        img_dataset = image.img_to_array(img_dataset)
        img_dataset = np.expand_dims(img_dataset, axis=0)
        img_dataset /= 255.
        #print(img)
        features_dataset.append((encoder.predict(img_dataset),img))
    
    pkl_filename =os.path.dirname(__file__) + '/../models/'+"DeepMethod_"+img_class + ".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(features_dataset, file)
    """
    pkl_filename =os.path.dirname(__file__) + '/../models/'+"DeepMethod_"+img_class + ".pkl"
    with open(pkl_filename, 'rb') as file:
        features_dataset = pickle.load(file)
    results = []
    # loop over our index
    for i in range(0, len(features_dataset)):
        d = euclidean(features_query, features_dataset[i][0][0])
        results.append((d, features_dataset[i][1]))
    # sort the results and grab the top ones
    results = sorted(results)[:10]
    print(results)

    fig=plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    plt.axis('off')
    f, axarr = plt.subplots(2, 5)
    idx=0
    f.suptitle('Deep Method Results')
    for i in range(rows):
        for j in range (columns):
            img = cv2.imread(dire+'\\'+results[idx][1])
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axarr[i,j].imshow(img)
            axarr[i,j].axis('off')
            axarr[i,j].set_title(idx+1)
            idx+=1

        

    

    plt.savefig(dirImgOut+'\\DeepMethod_result.jpg')

    