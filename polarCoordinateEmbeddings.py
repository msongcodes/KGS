'''
Created on 27 Mar 2022

@author: jack-kausch, based on the original by ejimenez-ruiz
'''
import requests
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#This script takes precomputed kg embeddings from DBpedia and uses principle component analysis
#to express each uri as a single polar coordinate vector.

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def getEmbeddings():
    
    #Check http://www.kgvec2go.org/
    
    print("\nPolar coordinate vector embedding for the resource 'Earth' (format: r, theta):")
    
    #http://dbpedia.org/resource/Chicago_Bulls
    kg_entity = "Pennsylvania"
    
    r = requests.get('http://www.kgvec2go.org/rest/get-vector/dbpedia/' + kg_entity) 
    uri = r.json()["uri"]
    vector_big = r.json()["vector"]
    
    #We duplicate the features here because the PCA library cannot handle a single feature
    axis = [0,1,2]

    #Load the vector in a dataframe
    dataframe = pd.DataFrame([vector_big],[axis])
    
    #Make sure the data has proper standard deviation before analysis
   # x = dataframe.loc[:, :].values
   # x = StandardScaler().fit_transform(x)
    
    #Take the principle components of the vector
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(dataframe)

    principal_components_df = pd.DataFrame(data = principal_components
             , columns = ['principal component 1', 'principal component 2'])

    vector_small = principal_components_df.iloc[0].tolist()

    #Convert the output into polar coordinate
    polar_vector = cart2pol(vector_small[0],vector_small[1])

    print(polar_vector)


  
    

#Query pre-computed knowledge graph embeddings
getEmbeddings()

print("\nTests successful!!")
