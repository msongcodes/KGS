'''
Created on 19 Jan 2021

@author: ejimenez-ruiz
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
    
    print("\nPolar coordinate vector embedding for the resource %kg_entity (format: r, theta):")
    
    #http://dbpedia.org/resource/Chicago_Bulls
    kg_entity = "Air"
    kg_entity2= "Earth"
    kg_entity3= "Fire"
    kg_entity4= "Water"
    
    
    r = requests.get('http://www.kgvec2go.org/rest/get-vector/dbpedia/' + kg_entity)
    print(r)
    r2 = requests.get('http://www.kgvec2go.org/rest/get-vector/dbpedia/' + kg_entity2)
    r3 = requests.get('http://www.kgvec2go.org/rest/get-vector/dbpedia/' + kg_entity3)
    r4 = requests.get('http://www.kgvec2go.org/rest/get-vector/dbpedia/' + kg_entity4)
    uri = r.json()["uri"]
    uri2 = r2.json()["uri"]
    uri3 = r3.json()["uri"]
    uri4 = r4.json()["uri"]
    v = r.json()["vector"]
    print(v)
    v2 = r2.json()["vector"]
    print(v2)
    v3 = r3.json()["vector"]
    v4 = r4.json()["vector"]
    vector_array=list(zip(v,v2,v3,v4))
    print(vector_array)
            
    
    #Load the vector in a dataframe
    dataframe = pd.DataFrame(vector_array).transpose()
    print(dataframe)
    
    #Make sure the data has proper standard deviation before analysis
   # x = dataframe.loc[:, :].values
   # x = StandardScaler().fit_transform(x)
    
    #Take the principle components of the vector
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(dataframe)

    principal_components_df = pd.DataFrame(data = principal_components
             , columns = ['principal component 1', 'principal component 2'])

    print(principal_components_df)
    vector_small = principal_components_df.iloc[0].tolist()

    #Convert the output into polar coordinate
    polar_vector = cart2pol(vector_small[0],vector_small[1])
    print(vector_small)
    print(polar_vector)


  
    

#Query pre-computed knowledge graph embeddings
getEmbeddings()

print("\nTests successful!!")
