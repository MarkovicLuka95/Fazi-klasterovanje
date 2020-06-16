import random
import pandas as pd
import sklearn.preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib




def fcm(podaci, broj_klastera=3, m=2, max_iter=30):
    #     #1)zadamo broj klastera, inicijalizujemo funkciju pripadnosti random
    #     funkcijaPripadnosti=inicijalizujMatricuPripadnosti(redovi,broj_klastera)
    #     #2)racunamo centre
    #     centroidi=racunanjeCentara(broj_klastera,redovi, m,podaci,funkcijaPripadnosti)
    #     #3)popravljamo funkciju pripadnosti
    #     funkcijaPripadnosti=AzuriranjeMatricePripadnosti(funkcijaPripadnosti,centroidi,podaci,m,redovi,broj_klastera)

    redovi = podaci.shape[0]
    trenutni = 0

    fig = plt.figure(figsize=(15, 9))
    plt_ind = 1

    for i in range(2,broj_klastera+1):
        funkcijaPripadnosti = inicijalizujMatricuPripadnosti(redovi, i)
        while trenutni < max_iter:
            centroidi = racunanjeCentara(i, redovi, m, podaci, funkcijaPripadnosti)
            funkcijaPripadnosti = AzuriranjeMatricePripadnosti(funkcijaPripadnosti, centroidi, podaci, m, redovi, i)
            trenutni += 1
        trenutni = 0

        fig.add_subplot(3, 3, plt_ind)

        for j in range (1,len(podaci)):
            plt.scatter(podaci["x"][j],podaci["y"][j], color = mesanjeBoja(i,funkcijaPripadnosti[j]), marker='o')
        for j in range(0, i):
            plt.scatter(centroidi.transpose()[0][j], centroidi.transpose()[1][j], color="black", marker='x')
        plt_ind +=1;
    plt.show()

    print(funkcijaPripadnosti)

def mesanjeBoja(broj_klastera,red):
    color = np.array([0.0,0.0,0.0])
    for i in range(broj_klastera):
        color +=colors[i]*red[i]
    return matplotlib.colors.to_hex(color)

def inicijalizujMatricuPripadnosti(red,broj_klastera):
    funkcijaPripadnosti=np.empty(shape=(red,broj_klastera))

    for i in range(red):
        random_num_list = [random.random() for i in range(broj_klastera)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        funkcijaPripadnosti[i]=temp_list;
    return funkcijaPripadnosti

def racunanjeCentara(broj_klastera, red, m,podaci,funkcijaPripadnosti):
    centroidi = np.empty(shape=(broj_klastera,2))
    broj_kordinata=2;

    for i in range(broj_klastera):
        for j in range (broj_kordinata):
            brojilac=0;
            imenilac=0;
            for k in range(red):
                brojilac+=podaci.iloc[k][j]*(funkcijaPripadnosti[k,i]**m)
                imenilac+=funkcijaPripadnosti[k,i]**m
            centroidi[i,j]=brojilac/imenilac
    return centroidi

def distance(dist, broj_klastera, red, centroidi, podaci):
    for i in range(broj_klastera):
        for k in range(red):
            dist[i][k] = np.linalg.norm(centroidi[i] - podaci.iloc[k])
    return dist


def AzuriranjeMatricePripadnosti(funkcijaPripadnosti, centroidi,podaci,m,red,broj_klastera):
    p = float(2/(m-1))
    dist=np.empty(shape=(broj_klastera,red))
    dist=distance(dist,broj_klastera,red,centroidi,podaci)

    for i in range(broj_klastera):
        for k in range(red):
            suma =  (dist[i][k]**(p))*sum(([1/((dist[j][k])**(p)) for j in range(broj_klastera)]))
            funkcijaPripadnosti[k,i]=1/suma

    return funkcijaPripadnosti


#pocetak meina

colors = np.array([[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[0,0,0],[0,1,0]])
broj_klastera=8
m=2
max_iter = 10
#df = pd.read_csv('data.csv')
df = pd.read_csv('data2.csv')
#df = pd.read_csv('data3.csv')
df = df.sample(200)
features = df.columns[0:2]
df1=df[['x','y']]

scaler = prep.MinMaxScaler().fit(df1)
x = pd.DataFrame(scaler.transform(df1))
x.columns = features

fcm(x,broj_klastera, m, max_iter)



