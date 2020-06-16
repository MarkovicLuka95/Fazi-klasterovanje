import random
import pandas as pd
import sklearn.preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib




def fcm(podaci, broj_klastera=3, m=2, max_iter=30):
    #     #1)zadamo broj klastera, inicijalizujemo funkciju pripadnosti random
    #     matricaPripadnosti=inicijalizujMatricuPripadnosti(brRedova,broj_klastera)
    #     #2)racunamo centre
    #     centroidi=racunanjeCentara(broj_klastera,brRedova, m,podaci,matricaPripadnosti)
    #     #3)popravljamo funkciju pripadnosti
    #     matricaPripadnosti=AzuriranjeMatricePripadnosti(matricaPripadnosti,centroidi,podaci,m,brRedova,broj_klastera)

    brRedova = podaci.shape[0]
    brIter = 0

    fig = plt.figure(figsize=(15, 9))
    plt_ind = 1

    for i in range(2,broj_klastera+1):
        matricaPripadnosti = inicijalizujMatricuPripadnosti(brRedova, i)
        while brIter < max_iter:
            centroidi = racunanjeCentara(i, brRedova, m, podaci, matricaPripadnosti)
            matricaPripadnosti = AzuriranjeMatricePripadnosti(matricaPripadnosti, centroidi, podaci, m, brRedova, i)
            brIter += 1
        brIter = 0

        fig.add_subplot(3, 3, plt_ind)

        for j in range (1,len(podaci)):
            plt.scatter(podaci["x"][j],podaci["y"][j], color = mesanjeBoja(i,matricaPripadnosti[j]), marker='o')
        for j in range(0, i):
            plt.scatter(centroidi.transpose()[0][j], centroidi.transpose()[1][j], color="black", marker='x')
        plt_ind +=1;
    plt.show()

    print(matricaPripadnosti)

def mesanjeBoja(broj_klastera,redUMatriciPripadsti):
    colors = np.array([[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[0,0,0],[0,1,0]])
    color = np.array([0.0,0.0,0.0])
    for i in range(broj_klastera):
        color +=colors[i]*redUMatriciPripadsti[i]
    return matplotlib.colors.to_hex(color)

def inicijalizujMatricuPripadnosti(brRedova,broj_klastera):
    matricaPripadnosti=np.empty(shape=(brRedova,broj_klastera))

    for i in range(brRedova):
        random_num_list = [random.random() for i in range(broj_klastera)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        matricaPripadnosti[i]=temp_list;
    return matricaPripadnosti

def racunanjeCentara(broj_klastera, brRedova, m,podaci,matricaPripadnosti):
    centroidi = np.empty(shape=(broj_klastera,2))
    broj_kordinata=2;

    for i in range(broj_klastera):
        for j in range (broj_kordinata):
            brojilac=0;
            imenilac=0;
            for k in range(brRedova):
                brojilac+=podaci.iloc[k][j]*(matricaPripadnosti[k,i]**m)
                imenilac+=matricaPripadnosti[k,i]**m
            centroidi[i,j]=brojilac/imenilac
    return centroidi

def racunanjeRastojanja(rastojanje, broj_klastera, brRedova, centroidi, podaci):
    for i in range(broj_klastera):
        for k in range(brRedova):
            rastojanje[i][k] = np.linalg.norm(centroidi[i] - podaci.iloc[k])
    return rastojanje


def AzuriranjeMatricePripadnosti(matricaPripadnosti, centroidi,podaci,m,brRedova,broj_klastera):
    p = float(2/(m-1))
    rastojanje=np.empty(shape=(broj_klastera,brRedova))
    rastojanje=racunanjeRastojanja(rastojanje,broj_klastera,brRedova,centroidi,podaci)

    for i in range(broj_klastera):
        for k in range(brRedova):
            suma =  (rastojanje[i][k]**(p))*sum(([1/((rastojanje[j][k])**(p)) for j in range(broj_klastera)]))
            matricaPripadnosti[k,i]=1/suma

    return matricaPripadnosti


#pocetak meina

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




