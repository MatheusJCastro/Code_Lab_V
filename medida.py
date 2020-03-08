################################################################
# Redução de Dados Física Experimental V - Efeito Fotoelétrico #
# Matheus J. Castro                                            #
# Version 5.2                                                  #
# Last Modification: 8 de Março de 2020                        #
################################################################

import numpy as np
import os
import matplotlib.pyplot as plt

cores = ["violeta", "azul", "verde", "amarelo", "vermelho"]
intensidade = [100, 80, 60, 40, 20]
dias = ["0503", "0603"]  # modifique essa lista colocando os dias que houve tomada de dados


# apenas fomato DDMM entre aspas duplas


def input_data():
    noise_bias = []
    noise_lamp = []
    final_data = {}

    for i in dias:
        noise_bias.append(np.loadtxt("{}_ruido_bias.csv".format(i), dtype=np.str, delimiter=";", skiprows=1))
        noise_lamp.append(np.loadtxt("{}_ruido_lampada_off.csv".format(i), dtype=np.str, delimiter=";", skiprows=1))

        data = {}
        for j in cores:
            par_data = {}
            for k in intensidade:
                arquivo = "{}_{}_{}.csv".format(i, j, k)
                if os.path.exists(arquivo):
                    par_data[k] = np.char.replace(np.loadtxt(arquivo, dtype=np.str, delimiter=";", skiprows=1),
                                                  ",", ".").astype(np.float64)
                    data[j] = par_data
            if len(data) != 0:
                final_data[i] = data

    noise_bias = np.char.replace(noise_bias, ",", ".").astype(np.float64)
    noise_lamp = np.char.replace(noise_lamp, ",", ".").astype(np.float64)

    return final_data, noise_bias, noise_lamp


def noise_removal(final_data, noise_bias, noise_lamp, bias=1, lamp=1):
    for i in dias:
        for j in cores:
            for k in intensidade:
                if i in final_data and j in final_data[i] and k in final_data[i][j]:
                    reduce = []
                    for m in range(len(noise_bias[0])):
                        reduce.append(final_data[i][j][k][m][1] - (bias * noise_bias[dias.index(i)][m][1] +
                                                                   lamp * noise_lamp[dias.index(i)][m][1]))
                    final_data[i][j][k] = [final_data[i][j][k].T[0], reduce]

    return final_data


def plot_all(data, meth_1, meth_3, save=False, show_meth_1=False, show_meth_3=False):
    color = {100: "black", 80: "blue", 60: "red", 40: "green", 20: "yellow"}
    if (len(cores) % 2) == 0:
        col = len(cores) / 2
    else:
        col = int((len(cores) - (len(cores) % 2)) / 2 + 1)

    plt.figure(figsize=(16, 10))

    for j in range(len(cores)):
        plt.subplot(2, col, j + 1)
        plt.xlabel("Tensão [V]")
        plt.ylabel("Corrente [nA]")
        plt.title(cores[j].capitalize())
        plt.xlim(-10, 10)
        plt.ylim(-1 * pow(10, -9), 2 * pow(10, -8))
        # for x in np.linspace(0, 1, 20):
        #    for y in np.linspace(0, 1, 20):
        #        plt.figtext(x, y, "Exemplo", fontsize=10, color='gray', alpha=.25, rotation=25)
        legenda = []
        for k in intensidade:
            for i in dias:
                if i in data and cores[j] in data[i] and k in data[i][cores[j]]:
                    x = np.array(data[i][cores[j]][k][0])
                    y = np.array(data[i][cores[j]][k][1])
                    legenda.append(plt.plot(x, y, "-", markersize=1, color=color[k], label="{}%".format(k)))
                    if show_meth_1:
                        plt.plot(meth_1["{}_{}_{}".format(i, cores[j], k)], 0, ".", markersize=10, color=color[k],
                                 label="Método 1")
        if show_meth_3:
            plt.plot(meth_3["{}".format(cores[j])], 0, "*", markersize=10, color="gray", label="Método 3")
        plt.legend(loc="upper left")
        plt.grid()

    if save:
        plt.savefig("Plot_of_all_data")
    plt.show()


def plot_same_intensity(data, meth_1, meth_3, k=100, save=False, show_meth_1=False, show_meth_3=False):
    color = {cores[0]: "violet", cores[1]: "blue", cores[2]: "green", cores[3]: "yellow", cores[4]: "red"}

    plt.figure(figsize=(8, 4.5))
    plt.xlabel("Tensão [V]")
    plt.ylabel("Corrente [nA]")
    plt.title("Todas as Frequências - {}%".format(k))
    plt.xlim(-10, 10)
    plt.ylim(-1 * pow(10, -9), 2 * pow(10, -8))
    # for x in np.linspace(0, 1, 12):
    #    for y in np.linspace(0, 1, 12):
    #        plt.figtext(x, y, "Exemplo", fontsize=10, color='gray', alpha=.5, rotation=25)
    for j in cores:
        for i in dias:
            if i in data and j in data[i] and k in data[i][j]:
                x = np.array(data[i][j][k][0])
                y = np.array(data[i][j][k][1])
                plt.plot(x, y, "-", markersize=1, color=color[j], label=j.capitalize())
                if show_meth_1:
                    plt.plot(meth_1["{}_{}_{}".format(i, j, k)], 0, ".", markersize=10, color=color[j])
        if show_meth_3:
            plt.plot(meth_3["{}".format(j)], 0, "*", markersize=10, color=color[j])
    plt.legend()
    plt.grid()

    if save:
        plt.savefig("Plot_same_intensity")
    plt.show()


def save_new_data(data):
    for i in dias:
        for j in cores:
            for k in intensidade:
                if i in data and j in data[i] and k in data[i][j]:
                    if not os.path.exists("reduced_data"):
                        os.mkdir("reduced_data")
                    np.savetxt("reduced_data/{}_{}_{}_reduced.csv".format(i, j, k), np.asarray(data[i][j][k]).T,
                               fmt="%s", delimiter=",")


def method_1(data, save=False):
    results = {}
    for i in dias:
        for j in cores:
            for k in intensidade:
                if i in data and j in data[i] and k in data[i][j]:
                    medida = np.copy(data[i][j][k]).tolist()
                    for m in range(len(medida[1])):
                        current = medida[1][m]
                        if current < 0:
                            medida[1][m] = -medida[1][m]
                    results["{}_{}_{}".format(i, j, k)] = medida[0][medida[1].index(min(medida[1]))]

    if save:
        if not os.path.exists("reduced_data"):
            os.mkdir("reduced_data")
        np.savetxt("reduced_data/1st_method_results.csv", list(results.items()), fmt="%s", delimiter=",")

    return results


def method_3(data, save=False):
    results = {}
    for j in cores:
        par_result = []
        for k in range(len(intensidade)):
            cur = True
            las = True
            current = []
            last = []
            while (cur is True) or (las is True):
                for i in dias:
                    if j in data[i] and intensidade[k] in data[i][j]:
                        current = data[i][j][intensidade[k]]
                        cur = False
                    if j in data[i] and intensidade[k - 1] in data[i][j]:
                        last = data[i][j][intensidade[k - 1]]
                        las = False
            compar = []
            for m in range(len(current[1])):
                if current[1][m] - last[1][m] < 0:
                    compar.append([current[0][m], -(current[1][m] - last[1][m])])
                else:
                    compar.append([current[0][m], current[1][m] - last[1][m]])
            compar = np.asarray(compar).T
            par_result.append(compar[0][list(compar[1]).index(min(compar[1]))])

            par_result.sort()
            if len(par_result) % 2 == 1:
                results["{}".format(j)] = par_result[len(par_result) // 2]
            elif len(par_result) % 2 == 0:
                results["{}".format(j)] = np.mean([par_result[len(par_result) // 2 - 1],
                                                   par_result[len(par_result) // 2]])
    if save:
        if not os.path.exists("reduced_data"):
            os.mkdir("reduced_data")
        np.savetxt("reduced_data/3st_method_results.csv", list(results.items()), fmt="%s", delimiter=",")

    return results


results_1 = {}
results_2 = {}
dt, noi_bias, noi_lamp = input_data()
reduced = noise_removal(dt, noi_bias, noi_lamp, bias=1, lamp=0)  # troque de 1 para 0 caso não vá remover algum dos
                                                                 # tipos de ruido
# save_new_data(reduced)  # descomente essa linha para salvar os dados no formato .csv
#results_1 = method_1(reduced, save=False)  # não está pronto
#results_2 = method_3(reduced, save=False)
#plot_all(reduced, results_1, results_2, save=False, show_meth_1=False, show_meth_3=False)  # descomente essa linha para plotar todos os dados
#plot_same_intensity(reduced, results_1, results_2, save=False, k=100, show_meth_1=False, show_meth_3=False)   # descomente essa linha para plotar um grafico com todas as
                                                                                            # frequencias em uma determinada intensidade (k)
                                                                                            # caso deseje salvar qualquer plot, troque False para True
