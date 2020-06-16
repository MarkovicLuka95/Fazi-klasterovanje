import pandas as pd
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


broj_klastera=9
#df = pd.read_csv('data.csv')
#df = pd.read_csv('data2.csv')
df = pd.read_csv('data3.csv')


df = df.sample(200)
features = df.columns[0:2]
df=df[['x','y']]


scaler = prep.MinMaxScaler().fit(df)
x = pd.DataFrame(scaler.transform(df))
x.columns = features



colors = ['red','green', 'yellow', 'blue','gray','orange','purple','brown','cyan']
fig = plt.figure(figsize=(15,9))
plt_ind=1

for i in range(2,broj_klastera+1):
    est = KMeans(n_clusters=i, init='random')
    est.fit(x)
    df['labels']= est.labels_

    centers = pd.DataFrame(scaler.inverse_transform(est.cluster_centers_),columns=features)
    print('centres', est.cluster_centers_)

    fig.add_subplot(3,3,plt_ind)

    for i in range(0, i):
        cluster = df.loc[df['labels'] == i]
        plt.scatter(cluster['x'], cluster['y'], color=colors[i - 1], label='Klaster %d' % i)
        plt.scatter(centers['x'], centers['y'], color='black', marker='x')

        plt.title('Senka koeficijent %0.3f' % silhouette_score(x, df['labels']))



    plt_ind += 1
plt.show()


