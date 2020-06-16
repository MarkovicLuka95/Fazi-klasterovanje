import random
import pandas as pd
import sklearn.preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def G_K(broj_klastera=3, m=2, max_iter=30, siroviPodaci=0):
    podaci = siroviPodaci.to_numpy()

    ro = np.ones(broj_klastera)
    brRedova = podaci.shape[0]
    brIter = 0
    fig = plt.figure(figsize=(15, 9))
    plt_ind = 1

    for i in range(2, broj_klastera + 1):
        matricaPripadnosti = inicijalizujMatricuPripadnosti(brRedova, i)
        while brIter < max_iter:
            centroidi = racunanjeCentara(i, brRedova, m, podaci, matricaPripadnosti)
            A = matricaA(brRedova, i, matricaPripadnosti, podaci, centroidi, ro)
            rastojanje = racunanjeRastojanja(i, brRedova, centroidi, podaci, A)
            matricaPripadnosti = AzuriranjeMatricePripadnosti(matricaPripadnosti, rastojanje, m, brRedova, i)
            brIter += 1
        brIter = 0;

        fig.add_subplot(3, 3, plt_ind)
        for j in range(1, len(podaci)):
            plt.scatter(siroviPodaci["x"][j], siroviPodaci["y"][j], color=mesanjeBoja(i, matricaPripadnosti[j]),
                        marker='o')

        for j in range(0, i):
            plt.scatter(centroidi.transpose()[0][j], centroidi.transpose()[1][j], color="black", marker='x')
        plt_ind += 1;
    plt.show()


def mesanjeBoja(broj_klastera, redMatricePripadnosti):
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
         [0, 0, 0], [0, 1, 0]])
    color = np.array([0.0, 0.0, 0.0])
    for i in range(broj_klastera):
        color += colors[i] * redMatricePripadnosti[i]
    return matplotlib.colors.to_hex(color)


def racunanjeRastojanja(broj_klastera, brRedova, centroidi, podaci, A):
    rastojanje = np.empty(shape=(broj_klastera, brRedova))
    for i in range(broj_klastera):
        for k in range(brRedova):
            diff = podaci[k] - centroidi[i]
            diff.reshape(1, 2)
            ret = np.matmul(diff, A[i])
            ret = np.matmul(ret, diff)
            rastojanje[i, k] = ret
    return rastojanje


def matricaUA(F, i, ro):
    A = pow(ro[i] * np.linalg.det(F), 1 / m) * np.linalg.inv(F)
    return A


def matricaA(brRedova, broj_klastera, matricaPripadnosti, podaci, centroidi, ro):
    f = np.empty(shape=(broj_klastera, broj_klastera))
    A = []
    for i in range(broj_klastera):
        brojilac = 0;
        imenilac = 0;
        for k in range(brRedova):
            a1 = (podaci[k] - centroidi[i])
            a1 = a1.reshape(1, 2)
            b = a1.reshape(-1, 1)
            e = np.matmul(b, a1)
            brojilac += (matricaPripadnosti[k, i] ** m) * e
            imenilac += matricaPripadnosti[k, i] ** m
        f = brojilac / imenilac
        Ai = matricaUA(f, i, ro)
        A.append(Ai)

    return A


def inicijalizujMatricuPripadnosti(brRedova, broj_klastera):
#radom marica sa sumom 1
    matricaPripadnosti = np.empty(shape=(brRedova, broj_klastera))
    for i in range(brRedova):
        random_num_list = [random.random() for i in range(broj_klastera)]
        summation = sum(random_num_list)
        temp_list = [x / summation for x in random_num_list]
        matricaPripadnosti[i] = temp_list;
    return matricaPripadnosti


def racunanjeCentara(broj_klastera, brRedova, m, data, matricaPripadnosti):
    centroidi = np.empty(shape=(broj_klastera, 2))
    broj_kordinata = 2;

    for i in range(broj_klastera):
        for j in range(broj_kordinata):
            brojilac = 0;
            imenilac = 0;
            for k in range(brRedova):
                brojilac += data[k][j] * (matricaPripadnosti[k, i] ** m)
                imenilac += matricaPripadnosti[k, i] ** m
            centroidi[i, j] = brojilac / imenilac
    return centroidi


def AzuriranjeMatricePripadnosti(matricaPripadnosti, rastojanje, m, brRedova, broj_klastera):
    p = float(1 / (m - 1))

    for i in range(broj_klastera):
        for k in range(brRedova):
            suma = (rastojanje[i][k] ** (p)) * sum(([1 / ((rastojanje[j][k]) ** (p)) for j in range(broj_klastera)]))
            matricaPripadnosti[k, i] = 1 / suma

    return matricaPripadnosti


# pocetak main fukcije

broj_klastera = 8
m = 2
max_iter = 20
# df = pd.read_csv('data.csv')
# df = pd.read_csv('data2.csv')
df = pd.read_csv('data3.csv')
df = df.sample(200)
features = df.columns[0:2]
df = df[['x', 'y']]

scaler = prep.MinMaxScaler().fit(df)
x = pd.DataFrame(scaler.transform(df[features]))
x.columns = features

G_K(broj_klastera, m, max_iter, x);



