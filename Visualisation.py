# %%
import matplotlib.pyplot as plt

# %%
import seaborn as sns; sns.set()  # for plot styling


# %%
import numpy as np


# %%
from sklearn.datasets import make_blobs


# %%
X, y_true = make_blobs(n_samples=300,centers=4, cluster_std=0.60, random_state=0)


# %%
plt.scatter(X[:, 0], X[:, 1], s=50);

# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# %%
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

# %%
from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)

# %%
labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');

# %%
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');

# %%
import pandas as pd


# %%
file = pd.read_csv('./ontology.embeddings.txt', header=None, sep=' ')
    

# %%
rep = list(file[0])

# %%
file = file.drop(columns=0)
print(file)

# %%
centroids = kmeans.cluster_centers_

# %%
from sklearn.decomposition import PCA


# %%
pca = PCA(n_components=2)

# %%
principal_components = pca.fit_transform(file)

# %%
principal_components_df = pd.DataFrame(data = principal_components
             , columns = ['x', 'y'], index = rep)

# %%
polardf = principal_components_df.drop(columns = 'x')
polardf = polardf.drop(columns = 'y')
principal_components_df = principal_components_df.drop(columns = 'phi')
principal_components_df = principal_components_df.drop(columns= 'ro')
print(polardf)

# %%
kmeans = KMeans(n_clusters=4).fit(principal_components_df)


# %%
kmeans2 = KMeans(n_clusters=4).fit(polardf)


# %%
#phi = np.arctan(principal_components_df.loc[:,"x"]*principal_components_df.loc[:,"y"])
#r   = np.power( np.power(principal_components_df.index,2) + np.power(principal_components_df.columns,2), 0.5 )

principal_components_df["phi"] = np.arctan(principal_components_df["x"]*principal_components_df["y"])
principal_components_df["ro"] = np.power( np.power(principal_components_df["x"],2) + np.power(principal_components_df["y"],2), 0.5 )


# %%
font = {
'size': 6}

plt.rc('font', **font)

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
theta = 2 * np.pi * np.random.rand(315)
#c = ax.scatter(polardf['phi'], polardf['ro'], c=theta,  s=(50), cmap='hsv', alpha=0.75)
plt.axis([-.4,1.3, -0.4, 0.4])
plt.rcParams["figure.figsize"] = (50,50)


#c=kmeans2.labels_.astype(float),

for k, v in principal_components_df.iterrows():
    plt.annotate(k, v)


# %%
print(principal_components_df)

# %%
centroids = kmeans.cluster_centers_


# %%
font = {
'size': 7}

plt.rc('font', **font)

#plt.scatter(principal_components_df['x'], principal_components_df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.axis([-0.5,1.3, -0.4, 0.4])
#plt.rcParams({'font.size': 12})

for k, v in principal_components_df.iterrows():
    plt.annotate(k, v)

x_data = principal_components_df['x']
y_data = principal_components_df['y']

txt_height = 0.04*(plt.ylim()[1] - plt.ylim()[0])
txt_width = 0.02*(plt.xlim()[1] - plt.xlim()[0])
text_positions = get_text_positions(x_data, y_data, txt_width, txt_height)
text_plotter(x_data, y_data, text_positions, ax, txt_width, txt_height)
plt.rcParams.update["figure.figsize"] = (50,50)

#plt.ylim(0,3610)
#plt.xlim(4.3,6.5)



# %%
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
#plt.annotate("Hello", (0.2, 0.2))


# %%
import string

# %%
file.drop(0, inplace=True, axis=1)

# %%
fig, ax = plt.subplots()

principal_components_df.plot('x', 'y', kind='scatter', ax=ax, c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.axis([1,1.1, 0.25, 0.35])

for k, v in principal_components_df.iterrows():
    ax.annotate(k, v)
    
fig.set_figheight(20)
fig.set_figwidth(20)


# %%
def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 1.5: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions


def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
    for x,y,t in list(zip(x_data, y_data, text_positions)):
        axis.text(x - .03, 1.02*t, '%d'%int(y),rotation=0, color='blue', fontsize=13)
        if y != t:
            axis.arrow(x, t+20,0,y-t, color='blue',alpha=0.2, width=txt_width*0.0,
                       head_width=.02, head_length=txt_height*0.5,
                       zorder=0,length_includes_head=True)

# %%



