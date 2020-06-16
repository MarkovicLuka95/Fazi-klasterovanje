import random
import pandas as pd
import sklearn.preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib




def G_K(podaci, broj_klastera=3, m=2, max_iter=30,siroviPodaci=0):

    #ne znam sta treba ro da mi bude, pise ||A|| = ro = konst, ali to me zbunjuje ako se A izracunava kasnije
    # negde kazu da se A random inicijalizuje...
    ro = np.ones(broj_klastera)
    redovi = podaci.shape[0]
    trenutni = 0
    fig = plt.figure(figsize=(15, 9))
    plt_ind = 1

    for i in range(2,broj_klastera+1):
        funkcijaPripadnosti = inicijalizujMatricuPripadnosti(redovi, i)
        while trenutni < max_iter:
            centroidi = racunanjeCentara(i, redovi, m, podaci, funkcijaPripadnosti)
            A = matricaKovarijanse(redovi,i,funkcijaPripadnosti,podaci,centroidi,ro)
            dist = distance(i,redovi,centroidi,podaci,A)
            funkcijaPripadnosti = AzuriranjeMatricePripadnosti(funkcijaPripadnosti, dist, m, redovi, i)
            trenutni += 1
        trenutni = 0;

        fig.add_subplot(3, 3, plt_ind)
        for j in range (1,len(podaci)):
            plt.scatter(siroviPodaci["x"][j],siroviPodaci["y"][j], color = mesanjeBoja(i,funkcijaPripadnosti[j]), marker='o')

        for j in range(0, i):
            plt.scatter(centroidi.transpose()[0][j], centroidi.transpose()[1][j], color="black", marker='x')
        plt_ind += 1;
    plt.show()

def mesanjeBoja(broj_klastera,red):
    color = np.array([0.0,0.0,0.0])
    for i in range(broj_klastera):
        color +=colors[i]*red[i]
    return matplotlib.colors.to_hex(color)



def distance(broj_klastera, red, centroidi, podaci,A):
    dist=np.empty(shape=(broj_klastera,red))
    for i in range(broj_klastera):
        for k in range (red):
            diff = podaci[k]-centroidi[i]
            diff.reshape(1,2)
            ret = np.matmul(diff, A[i])
            ret = np.matmul(ret,diff)
            dist[i,k]=ret
    return dist

def matricaA(F,i, ro):
    A= pow(ro[i]*np.linalg.det(F), 1/m) * np.linalg.inv(F)
    return A

def matricaKovarijanse(red,broj_klastera,funkcijaPripadnosti,podaci,centroidi,ro):

    f=np.empty(shape=(broj_klastera,broj_klastera))
    A=[]
    for i in range(broj_klastera):
        brojilac = 0;
        imenilac = 0;
        for k in range(red):
            a1=(podaci[k] - centroidi[i])
            a1=a1.reshape(1,2)
            b=a1.reshape(-1,1)
            e = np.matmul(b,a1)
            brojilac += (funkcijaPripadnosti[k, i] ** m) * e
            imenilac += funkcijaPripadnosti[k, i] ** m
        f=brojilac/imenilac
        a=matricaA(f,i,ro)
        A.append(a)

    return A


def inicijalizujMatricuPripadnosti(red,broj_klastera):
    funkcijaPripadnosti=np.empty(shape=(red,broj_klastera))
    for i in range(red):
        random_num_list = [random.random() for i in range(broj_klastera)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        funkcijaPripadnosti[i]=temp_list;
    return funkcijaPripadnosti

# #ovo izgleda ne treba....
# def initCentersandA(red,broj_klastera):
#     centroidi=np.empty(shape=(broj_klastera,2))
#     for i in range(broj_klastera):
#         random_num_list = [random.random() for i in range(2)]
#         centroidi[i]=random_num_list;
#     return (centroidi)

def racunanjeCentara(broj_klastera, red, m,data,funkcijaPripadnosti):
    centroidi = np.empty(shape=(broj_klastera,2))
    broj_kordinata=2;

    for i in range(broj_klastera):
        for j in range (broj_kordinata):
            brojilac=0;
            imenilac=0;
            for k in range(red):
                brojilac+=data[k][j]*(funkcijaPripadnosti[k,i]**m)
                imenilac+=funkcijaPripadnosti[k,i]**m
            centroidi[i,j]=brojilac/imenilac
    return centroidi




def AzuriranjeMatricePripadnosti(funkcijaPripadnosti, dist,m,red,broj_klastera):
    p = float(1/(m-1))

    for i in range(broj_klastera):
        for k in range(red):
            suma =  (dist[i][k]**(p))*sum(([1/((dist[j][k])**(p)) for j in range(broj_klastera)]))
            funkcijaPripadnosti[k,i]=1/suma


    return funkcijaPripadnosti


#pocetak meina

colors = np.array([[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[0,0,0],[0,1,0]])
broj_klastera=8
m=2
max_iter = 20
#df = pd.read_csv('data.csv')
#df = pd.read_csv('data2.csv')
df = pd.read_csv('data3.csv')
df = df.sample(300)
features = df.columns[0:2]
df=df[['x','y']]

scaler = prep.MinMaxScaler().fit(df)
x = pd.DataFrame(scaler.transform(df[features]))
x.columns = features
siroviPodaci=x
x=x.to_numpy()

G_K(x,broj_klastera, m, max_iter,siroviPodaci);



