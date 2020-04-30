################################################################
# Redução de Dados Física Experimental V - Efeito Fotoelétrico #
# Matheus J. Castro                                            #
# Version 8.74                                                 #
# Last Modification: 27 de Março de 2020                       #
################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk

window = tk.Tk()
width = 1000
height = 500
# posRight = int(window.winfo_screenwidth() / 2 - width / 2)
posRight = 300
posDown = int(window.winfo_screenheight() / 2 - height / 2)
window.geometry("{}x{}+{}+{}".format(int(width), int(height), posRight, posDown))
window.title("Efeito Fotoelétrico por Matheus J. Castro")

cores = ["violeta", "azul", "verde", "amarelo", "vermelho"]
intensidade = [100, 80, 60, 40, 20]
dias = ["1003"]  # modifique essa lista colocando os dias que houve tomada de dados"0503", "0603",
# apenas fomato DDMM entre aspas duplas


def input_data():
    noise_bias = []
    noise_lamp = []
    final_data = {}
    repete = range(1, repeat_op.get() + 1)

    warning = "Ruídos não encontrados:"
    if not os.path.exists("raw_data"):
        log.config(text="Pasta raw_data não encontrada.")
        return
    for i in dias:
        if bias.get() and os.path.exists("raw_data/{}_ruido_bias.csv".format(i)):
            noise_bias.append(np.loadtxt("raw_data/{}_ruido_bias.csv".format(i), dtype=np.str, delimiter=";", skiprows=1))
            noise_bias[-1] = np.char.replace(noise_bias[-1], ",", ".").astype(np.float64)
        else:
            noise_bias.append([[0, 0]])
            if bias.get():
                warning = warning + "\nBias {}".format(i)
        if lamp.get() and os.path.exists("raw_data/{}_ruido_lampada_off.csv".format(i)):
            noise_lamp.append(np.loadtxt("raw_data/{}_ruido_lampada_off.csv".format(i), dtype=np.str, delimiter=";", skiprows=1))
            noise_lamp[-1] = np.char.replace(noise_lamp[-1], ",", ".").astype(np.float64)
        else:
            noise_lamp.append([[0, 0]])
            if lamp.get():
                warning = warning + "\nLamp {}".format(i)

        data = {}
        for j in cores:
            par_data = {}
            for k in intensidade:
                repete_data = {}
                for l in repete:
                    arquivo = "raw_data/{}_{}_{}_{}.csv".format(i, j, k, l)
                    if os.path.exists(arquivo):
                        repete_data[l] = np.char.replace(np.loadtxt(arquivo, dtype=np.str, delimiter=";", skiprows=1),
                                                         ",", ".").astype(np.float64)
                if len(repete_data) != 0:
                    par_data[k] = repete_data
            if len(par_data) != 0:
                data[j] = par_data
        if len(data) != 0:
            final_data[i] = data
    if warning == "Ruídos não encontrados:":
        warning = ""
    if len(final_data) == 0:
        warning += "\nNenhum arquivo de dado encontrado."
    log.config(text=warning)

    return final_data, noise_bias, noise_lamp


def noise_removal(final_data, noise_bias, noise_lamp, red_bias=1, red_lamp=1, sig=False):
    if sig:
        repete = range(1, repeat_op_sig.get() + 1)
    else:
        repete = range(1, repeat_op.get() + 1)

    for i in dias:
        for j in cores:
            for k in intensidade:
                for l in repete:
                    if i in final_data and j in final_data[i] and k in final_data[i][j] and l in final_data[i][j][k]:
                        reduce = []
                        for m in range(len(final_data[i][j][k][l])):
                            if red_bias == 1:
                                red_bias = noise_bias[dias.index(i)][m][1]
                            if red_lamp == 1:
                                red_lamp = noise_lamp[dias.index(i)][m][1]
                            reduce.append(final_data[i][j][k][l][m][1] - (red_bias + red_lamp))
                        final_data[i][j][k][l] = [final_data[i][j][k][l].T[0], reduce]

    return final_data


def plot_all(data, meth_1, meth_3, show=True, save=False, show_meth_1=False, show_meth_3=False):
    color = {100: "black", 80: "blue", 60: "red", 40: "green", 20: "yellow"}
    color_bar = {100: "gray", 80: "cyan", 60: "salmon", 40: "lightgreen", 20: "lightyellow"}
    if (len(cores) % 2) == 0:
        col = len(cores) / 2
    else:
        col = int((len(cores) - (len(cores) % 2)) / 2 + 1)

    maxi = []
    for i in dias:
        for j in cores:
            if i in data and j in data[i] and 100 in data[i][j]:
                maxi.append(np.max(data[i][j][100][1]))
    maxi = max(maxi)
    exp = 0
    while maxi < 1:
        maxi = maxi*10
        exp += 1
    if int(maxi) < maxi:
        maxi = int(maxi) + 1
    else:
        maxi = int(maxi)

    plt.figure(figsize=(16*2, 9*2))

    for j in range(len(cores)):
        plt.subplot(2, col, j + 1)
        plt.xlabel("Tensão [V]")
        plt.ylabel("Corrente [nA]")
        plt.title(cores[j].capitalize())
        plt.xlim(-10, 10)
        if scale_plot1_op.get():
            plt.ylim(-1 * pow(10, -9), maxi * pow(10, -exp))
        # for x in np.linspace(0, 1, 20):
        #    for y in np.linspace(0, 1, 20):
        #        plt.figtext(x, y, "Exemplo", fontsize=10, color='gray', alpha=.25, rotation=25)
        for k in intensidade:
            for i in dias:
                if i in data and cores[j] in data[i] and k in data[i][cores[j]]:
                    x = np.array(data[i][cores[j]][k][0])
                    y = np.array(data[i][cores[j]][k][1])
                    y_erro = np.array(data[i][cores[j]][k][2])
                    plt.plot(x, y, "-", markersize=1, color=color[k], label="{}%".format(k), zorder=2)
                    plt.errorbar(x, y, yerr=y_erro, color=color_bar[k], zorder=1)
                    if show_meth_1 and "{}_{}_{}".format(i, cores[j], k) in meth_1:
                        plt.plot(meth_1["{}_{}_{}".format(i, cores[j], k)][0], 0, ".", markersize=10, color=color[k],
                                 label="Método 1", zorder=3)
                        plt.errorbar(meth_1["{}_{}_{}".format(i, cores[j], k)][0], 0,
                                     xerr=meth_1["{}_{}_{}".format(i, cores[j], k)][1], color=color_bar[k], zorder=3)
        if show_meth_3:
            plt.plot(meth_3["{}".format(cores[j])][0], 0, "*", markersize=10, color="gray", label="Método 3", zorder=4)
            plt.errorbar(meth_3["{}".format(cores[j])][0], 0,
                         xerr=meth_3["{}".format(cores[j])][1], color="gray", zorder=4)
        plt.legend(loc="upper left")
        plt.grid()

    if save:
        if not os.path.exists("plots"):
            os.mkdir("plots")
        if scale_plot1_op.get():
            plt.savefig("plots/Plot_of_all_data_scale")
        else:
            plt.savefig("plots/Plot_of_all_data_outof_scale")
    if show:
        plt.show()
    else:
        plt.close()


def plot_same_intensity(data, meth_1, meth_3, k=100, show=True, save=False, show_meth_1=False, show_meth_3=False):
    color = {cores[0]: "violet", cores[1]: "blue", cores[2]: "green", cores[3]: "yellow", cores[4]: "red"}

    plt.figure(figsize=(16, 9))
    plt.xlabel("Tensão [V]")
    plt.ylabel("Corrente [nA]")
    plt.title("Todas as Frequências - {}%".format(k))
    plt.xlim(-10, 10)
    # for x in np.linspace(0, 1, 12):
    #    for y in np.linspace(0, 1, 12):
    #        plt.figtext(x, y, "Exemplo", fontsize=10, color='gray', alpha=.5, rotation=25)
    for j in cores:
        for i in dias:
            if i in data and j in data[i] and k in data[i][j]:
                x = np.array(data[i][j][k][0])
                y = np.array(data[i][j][k][1])
                y_erro = np.array(data[i][j][k][2])
                plt.plot(x, y, "-", markersize=2, color=color[j], label=j.capitalize(), zorder=2)
                plt.errorbar(x, y, yerr=y_erro, color=color[j], zorder=1)
                if show_meth_1 and "{}_{}_{}".format(i, j, k) in meth_1:
                    plt.plot(meth_1["{}_{}_{}".format(i, j, k)][0], 0, ".", markersize=10, color=color[j],
                             label="Método 1", zorder=3)
                    plt.errorbar(meth_1["{}_{}_{}".format(i, j, k)][0], 0,
                                 xerr=meth_1["{}_{}_{}".format(i, j, k)][1], color=color[j], zorder=3)
        if show_meth_3:
            plt.plot(meth_3["{}".format(j)][0], 0, "*", markersize=10, color=color[j], zorder=4)
            plt.errorbar(meth_3["{}".format(j)][0], 0, xerr=meth_3["{}".format(j)][1], color=color[j], zorder=4)
    plt.legend()
    plt.grid()

    if save:
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.savefig("plots/Plot_same_intensity_{}".format(k))
    if show:
        plt.show()
    else:
        plt.close()


def save_new_data(data):
    head = "Tensão [V], Corrente [A], Incerteza Corrente [A]"
    for i in dias:
        for j in cores:
            for k in intensidade:
                if i in data and j in data[i] and k in data[i][j]:
                    if not os.path.exists("reduced_data"):
                        os.mkdir("reduced_data")
                    save_data = np.copy(data[i][j][k])
                    if op_sig.get() == "Mínimos e Máximos":
                        save_data = [data[i][j][k][0], data[i][j][k][1], data[i][j][k][2][0], data[i][j][k][2][1]]
                        head = "Tensão [V], Corrente [A], Incerteza Corrente Inferior [A], " \
                               "Incerteza Corrente Superior [A]"
                    np.savetxt("reduced_data/{}_{}_{}_1_reduced.csv".format(i, j, k), np.asarray(save_data).T,
                               fmt="%s", delimiter=",", header=head)


def sigma_1(data, bia, lam):
    repete = range(1, repeat_op_sig.get() + 1)
    # verificar pasta
    if not os.path.exists("sigma"):
        log.config(text="Pasta sigma não encontrada.")
        return False
    # verficar os arquivos e achar a cor
    cor = ""
    for i in dias:
        for j in cores:
            for k in intensidade:
                for l in repete:
                    if os.path.exists("sigma/{}_{}_{}_{}.csv".format(i, j, k, l)):
                        if len(cor) == 0:
                            cor = j
                        elif cor != j:
                            log.config(text="Arquivos em \"sigma\" não conferem.\nIncerteza não encontrada.")
                            return False
    # import dos arquivos
    final_data = {}
    dia = ""
    inten = ""
    for i in dias:
        par_data = {}
        for k in intensidade:
            repete_data = {}
            for l in repete:
                arquivo = "sigma/{}_{}_{}_{}.csv".format(i, cor, k, l)
                if os.path.exists(arquivo):
                    dia = i
                    inten = k
                    repete_data[l] = np.char.replace(np.loadtxt(arquivo, dtype=np.str, delimiter=";", skiprows=1),
                                                     ",", ".").astype(np.float64)
            if len(repete_data) != 0:
                par_data[k] = repete_data
        if len(par_data) != 0:
            final_data[i] = par_data
    # remove ruido dos arquivos
    final_data = noise_removal(final_data, bia, lam, red_bias=bias.get(), red_lamp=lamp.get(), sig=True)
    # acha a incerteza
    compar_data = {}
    for k in intensidade:
        par_data = []
        for m in range(len(final_data[dia][inten][1])):
            compar = []
            for l in repete:
                for i in dias:
                    if i in final_data and k in final_data[i] and l in final_data[i][k]:
                        dado = final_data[i][k][l][m][1]
                        compar.append(dado)
            uncer = 0
            if op_sig.get() == "Desvio Padrão":
                uncer = np.std(compar)
            elif op_sig.get() == "Mínimos e Máximos":
                val = np.median(compar)
                uncer = [val - np.min(compar), np.max(compar) - val]  # minimos e maximos como incerteza
            elif op_sig.get() == "Mediana dos Mínimos e Máximos":
                val = np.median(compar)
                uncer = np.median([val - np.min(compar), np.max(compar) - val])  # mediana dos min e maxi como incerteza
            par_data.append(uncer)

        if len(par_data) != 0:
            if op_sig.get() == "Mínimos e Máximos":
                compar_data[k] = np.asarray(par_data).T  # se usar min e max como incerteza, colocar a transformada
            else:
                compar_data[k] = np.asarray(par_data)

    sig_data = {}
    for j in cores:
        sig_data[j] = compar_data

    data = add_sig(data, sig_data)

    return data


def sigma_2(data):
    repete = range(1, repeat_op.get() + 1)

    len_m = 0
    for i in dias:
        for j in cores:
            for k in intensidade:
                for l in repete:
                    if i in data and j in data[i] and k in data[i][j] and l in data[i][j][k]:
                        len_m = len(data[i][j][k][l][1])
                        break
                if len_m != 0:
                    break
            if len_m != 0:
                break
        if len_m != 0:
            break

    compar_data = {}
    for j in cores:
        k_par_data = {}
        for k in intensidade:
            par_data = []
            for m in range(len_m):
                compar = []
                for l in repete:
                    for i in dias:
                        if i in data and j in data[i] and k in data[i][j] and l in data[i][j][k]:
                            dado = data[i][j][k][l][1][m]
                            compar.append(dado)
                uncer = 0
                if op_sig.get() == "Desvio Padrão":
                    uncer = np.std(compar)
                elif op_sig.get() == "Mínimos e Máximos":
                    val = np.median(compar)
                    uncer = [val - np.min(compar), np.max(compar) - val]  # minimos e maximos como incerteza
                elif op_sig.get() == "Mediana dos Mínimos e Máximos":
                    val = np.median(compar)
                    uncer = np.median([val - np.min(compar), np.max(compar) - val])
                    # mediana dos min e maxi como incerteza
                par_data.append(uncer)

            if len(par_data) != 0:
                if op_sig.get() == "Mínimos e Máximos":
                    k_par_data[k] = np.asarray(par_data).T  # se usar min e max como incerteza, colocar a transformada
                else:
                    k_par_data[k] = np.asarray(par_data)
        if len(k_par_data) != 0:
            compar_data[j] = k_par_data

    data = add_sig(data, compar_data)

    return data


def add_sig(data, sig_data):
    repete = range(1, repeat_op.get() + 1)

    for j in cores:
        for k in intensidade:
            volt = []
            dia = ""
            compar = []
            for l in repete:
                for i in dias:
                    if i in data and j in data[i] and k in data[i][j] and l in data[i][j][k]:
                        dia = i
                        dado = data[i][j][k][l][1]
                        compar.append(dado)
                        volt = data[i][j][k][l][0]
                        del data[i][j][k][l]
            par_data = []
            if len(compar) != 0:
                for m in range(len(compar[0])):
                    med_data = []
                    for n in range(len(compar)):
                        med_data.append(compar[n][m])
                    par_data.append(np.median(med_data))
            if len(par_data) != 0:
                data[dia][j][k] = [volt, par_data, sig_data[j][k]]

    return data


def method_1(data, save=False):
    results = {}
    for i in dias:
        for j in cores:
            for k in intensidade:
                if i in data and j in data[i] and k in data[i][j]:
                    medida = np.copy(data[i][j][k]).tolist()
                    m = len(medida[1])-1
                    val = None
                    inc_val = None
                    if medida[1][len(medida[1]) - 1] > 0:
                        while m >= 0:
                            if medida[1][m] <= 0:
                                X = np.abs(medida[0][m+1] - medida[0][m])
                                y_a = np.abs(medida[1][m])
                                y_b = np.abs(medida[1][m+1])
                                x_a = (y_a * X)/(y_a + y_b)
                                val = medida[0][m] + x_a
                                sig_ya = np.average(np.asarray(medida[2]).T[m])
                                sig_yb = np.average(np.asarray(medida[2]).T[m+1])
                                inc_val = x_a * np.sqrt((pow((sig_ya/y_a), 2)*(y_b/(y_a+y_b))) +
                                                        ((pow(sig_ya, 2)+pow(sig_yb, 2))/pow((y_a+y_b), 2)))
                                break
                            m -= 1
                    if val is not None:
                        results["{}_{}_{}".format(i, j, k)] = val, inc_val
    if save:
        head = "Curva, V0 [V], Incerteza V0 [V]"

        if not os.path.exists("methods_results"):
            os.mkdir("methods_results")

        results_array = np.asarray(list(results.items()))
        save_results = []
        for i in range(len(results_array)):
            save_results.append([results_array[i][0], results_array[i][1][0], results_array[i][1][1]])

        np.savetxt("methods_results/1st_method_results.csv", save_results, fmt="%s", delimiter=",", header=head)

    return results


def method_3(data, save=False, save_residual=False, main_iten=100, sec_iten=80, cor=cores[0]):
    results = {}
    for j in cores:
        val_list = []
        inc_list = []
        for k in range(len(intensidade)):
            int_val_list = []
            int_inc_list = []
            for m in range(k+1, len(intensidade)):
                main_array = np.asarray([])
                sec_array = np.asarray([])
                for i in dias:
                    if i in data and j in data[i] and intensidade[k] in data[i][j]:
                        main_array = np.copy(data[i][j][intensidade[k]])
                    if i in data and j in data[i] and intensidade[m] in data[i][j]:
                        sec_array = np.copy(data[i][j][intensidade[m]])
                if len(main_array) != 0 and len(sec_array) != 0:
                    dif = np.asarray(main_array[1]) - np.asarray(sec_array[1])
                    n = len(dif)-1
                    val = None
                    inc_val = None
                    if dif[len(dif) - 1] > 0:
                        while n >= 0:
                            if dif[n] <= 0:
                                X = np.abs(main_array[0][n + 1] - main_array[0][n])
                                y_a = np.abs(dif[n])
                                y_b = np.abs(dif[n + 1])
                                x_a = (y_a * X) / (y_a + y_b)
                                val = main_array[0][n] + x_a
                                sig_yak = np.average(np.asarray(main_array[2]).T[n])
                                sig_ybk = np.average(np.asarray(main_array[2]).T[n + 1])
                                sig_yam = np.average(np.asarray(sec_array[2]).T[n])
                                sig_ybm = np.average(np.asarray(sec_array[2]).T[n + 1])
                                sig_ya = np.sqrt(pow(sig_yak, 2)+pow(sig_yam, 2))
                                sig_yb = np.sqrt(pow(sig_ybk, 2) + pow(sig_ybm, 2))
                                inc_val = x_a * np.sqrt((pow((sig_ya / y_a), 2) * (y_b / (y_a + y_b))) +
                                                        ((pow(sig_ya, 2) + pow(sig_yb, 2)) / pow((y_a + y_b), 2)))
                                break
                            n -= 1
                    if val is not None:
                        int_val_list.append(val)
                        int_inc_list.append(inc_val)
                    if save_residual and intensidade[k] == main_iten and intensidade[m] == sec_iten and j == cor:
                        plt.figure(figsize=(16, 9))
                        plt.title("Gráfico de Residuos\nDiferença entre duas intensidades de uma mesma cor.")
                        plt.plot(main_array[0], dif, ".", markersize=5, label="Diferença Pontos")
                        plt.plot(main_array[0], dif, "-", markersize=0.5, label="Diferença Reta")
                        plt.plot([-10, 10], [0, 0])
                        plt.plot(val, 0, ".", markersize=10, label="Valor Encontrado")
                        plt.legend()
                        plt.grid()
                        plt.savefig("Plot_Residual")
                        plt.show()
                        plt.close()
            if len(int_val_list) != 0:
                val_list.append(np.median(int_val_list))
                inc_list.append(np.median(int_inc_list))
        if len(val_list) != 0:
            results[j] = np.median(val_list), np.median(inc_list)

    if save:
        head = "Cor, V0 [V], Incerteza V0 [V]"

        if not os.path.exists("methods_results"):
            os.mkdir("methods_results")

        results_array = np.asarray(list(results.items()))
        save_results = []
        for i in range(len(results_array)):
            save_results.append([results_array[i][0], results_array[i][1][0], results_array[i][1][1]])

        np.savetxt("methods_results/3rd_method_results.csv", save_results, fmt="%s", delimiter=",", header=head)

    return results


def get_lambda():
    if os.path.exists("cores.txt"):
        cor = np.loadtxt("cores.txt", dtype=np.str, delimiter=";")
    else:
        log.config(text="Arquivo \"cores.txt\" não encontrado.")
        return
    cor.T[0] = np.char.replace(cor.T[0], " ", "")
    cor.T[1] = np.char.replace(cor.T[1], ",", ".").astype(np.float64)
    if len(cor.T) == 4:
        cor.T[2] = np.char.replace(cor.T[2], ",", ".").astype(np.float64)
        cor.T[3] = np.char.replace(cor.T[3], ",", ".").astype(np.float64)
    for m in range(len(cor.T[0])):
        cor.T[0][m] = cor.T[0][m].lower()
    dic_cor = {}
    inc_cor = {}
    for m in range(len(cor.T[0])):
        if cor.T[0][m] not in dic_cor:
            dic_cor[cor.T[0][m]] = float(cor.T[1][m])
            if len(cor.T) == 4:
                inc_cor[cor.T[0][m]] = [float(cor.T[1][m]) - float(cor.T[2][m]),
                                        float(cor.T[3][m]) - float(cor.T[1][m])]
        else:
            dic_value = float(dic_cor[cor.T[0][m]])
            current_value = float(cor.T[1][m])
            dic_cor[cor.T[0][m]] = np.average([dic_value, current_value])
            if len(cor.T) == 4:
                inc_value = [float(inc_cor[cor.T[0][m]][0]), float(inc_cor[cor.T[0][m]][1])]
                current_inc = [float(cor.T[1][m]) - float(cor.T[2][m]), float(cor.T[3][m]) - float(cor.T[1][m])]
                inc_cor[cor.T[0][m]] = [np.average(inc_value[0], current_inc[0]),
                                        np.average(inc_value[1], current_inc[1])]
        if len(cor.T) == 2:
            inc_cor[cor.T[0][m]] = [0, 0]
    return dic_cor, inc_cor


def func_adjs(x, y, degree=1):
    ajuste_polinomial_Q2 = np.polyfit(x, y, degree)
    func = np.poly1d(ajuste_polinomial_Q2)
    phi = func(0)
    h = func(1) - phi
    func_x = np.linspace(x[0], x[-1], 5)
    func_y = func(func_x)

    return func_x, func_y, h, phi


def planck_meth(result, show=False, save=False):
    eV = 1.602176634E-19  # J
    m_nm = 1E-9  # nm
    c = 299792458  # m/s

    comp_cor = get_lambda()[0]
    inc_comp_cor = get_lambda()[1]

    if meth_planck.get() == 1:
        par_result = {}
        for j in cores:
            cor_par_result = []
            for k in intensidade:
                for i in dias:
                    if "{}_{}_{}".format(i, j, k) in result:
                        cor_par_result.append(result["{}_{}_{}".format(i, j, k)])
            if len(cor_par_result) != 0:
                cor_par_result = np.asarray(cor_par_result).T
                cor_par_result = [np.median(cor_par_result[0]), np.median(cor_par_result[1])]
                par_result[j] = cor_par_result
        result = par_result

    plot_lambda = []
    plot_result = []
    plot_error_x = []
    plot_error_y = []
    plot_lambda_min = []
    plot_lambda_max = []
    plot_result_max = []
    plot_result_min = []
    for j in cores:
        if j in result and j != "vermelho":
            if j not in comp_cor:
                log.config(text="Arquivo \"cores.txt\" não \ncorresponde com os dados.")
                return
            else:
                #abs_result = np.abs(result[j][0])
                abs_result = result[j][0]
                plot_lambda.append(c/(comp_cor[j]*m_nm))
                plot_result.append(abs_result*eV)
                plot_error_y.append(result[j][1]*eV)
                plot_result_max.append((abs_result + result[j][1])*eV)
                plot_result_min.append((abs_result - result[j][1])*eV)

                form = c/pow(comp_cor[j]*m_nm, 2)
                error_x = [form*inc_comp_cor[j][0]*m_nm, form*inc_comp_cor[j][1]*m_nm]
                plot_error_x.append(error_x)

                plot_lambda_max.append(c/(comp_cor[j]*m_nm) + error_x[1])
                plot_lambda_min.append(c/(comp_cor[j]*m_nm) - error_x[0])
    plot_error_x = np.asarray(plot_error_x).T

    func_x_max, func_y_max, h_max, phi_max = func_adjs(plot_lambda_max, plot_result_max, degree=1)
    func_x_min, func_y_min, h_min, phi_min = func_adjs(plot_lambda_min, plot_result_min, degree=1)
    func_x, func_y, h, phi = func_adjs(plot_lambda, plot_result, degree=1)

    inc_h = np.abs(np.average([h_max - h, h - h_min]))
    inc_phi = np.abs(np.average([phi_max - phi, phi - phi_min])/eV)
    phi = np.absolute(phi)/eV

    print("Método {}: Valor de h={:.2e} | Valor de phi={:.2f}".format(meth_planck.get(), h, phi))
    print("-" * 52)
    for j in cores:
        print("{:<8} | f={:.2e} | h*f={:.2e} | e*phi={:.2e} | Faz efeito? {}."
              .format(j.capitalize(), c/(comp_cor[j]*m_nm), (h*c)/(comp_cor[j]*m_nm), eV*phi,
                      (h*c)/(comp_cor[j]*m_nm) > eV*phi))
    print("-"*70)

    plt.figure(figsize=(16, 9))
    plt.xlabel("Frequência (c/\u03bb) [Hz]")
    plt.ylabel("Energia (eVo) [J]")
    plt.title("Energia x Comprimento de Onda")
    plt.xlim(min(func_x_min)*0.99, max(func_x_max)*1.01)
    # for x in np.linspace(0, 1, 12):
    #    for y in np.linspace(0, 1, 12):
    #        plt.figtext(x, y, "Exemplo", fontsize=10, color='gray', alpha=.5, rotation=25)
    plt.errorbar(plot_lambda, plot_result, xerr=plot_error_x, yerr=plot_error_y,
                 fmt=".", markersize=10, label="eVo por f")
    plt.plot(func_x, func_y, "-", label="Função ajustada")
    plt.plot(func_x_max, func_y_max, "-", label="Função ajustada superior")
    plt.plot(func_x_min, func_y_min, "-", label="Função ajustada inferior")
    plt.figtext(0.3, 0.834, "h = {:.2e}+/-{:.1e} m\u00b2kg/s\n\u03d5 = {:.2f}+/-{:.2f} V"
                .format(h, inc_h, phi, inc_phi), bbox=dict(facecolor="cyan", alpha=1))
    plt.legend(loc="upper left")
    plt.grid()

    if save:
        if not os.path.exists("planck_results"):
            os.mkdir("planck_results")
        plt.savefig("planck_results/Plot_Planck_method_{}".format(meth_planck.get()))

        head = "Frequência [Hz], Incerteza Frequência Min [Hz], Incerteza Frequência Max [Hz], Energia (eVo) [J], " \
               "Incerteza Energia [J]"
        save_results = np.asarray([plot_lambda, plot_error_x[0], plot_error_x[1], plot_result, plot_error_y]).T
        np.savetxt("planck_results/{}_method_plot.csv".format(meth_planck.get()), save_results,
                   fmt="%.2e", delimiter=",", header=head)
        head = " , Constante de Planck [m\u00b2kg/s], \u03d5 [V]"
        save_results = np.asarray([["Valor", "Incerteza"], [h, inc_h], [phi, inc_phi]]).T
        np.savetxt("planck_results/{}_method_results.csv".format(meth_planck.get()), save_results,
                   fmt="%s", delimiter=",", header=head)
    if show:
        plt.show()
    else:
        plt.close()


def check():
    global reduced_status
    global calc_1, calc_2, calc_3, calc_4

    reduced_status = False
    calc_1 = False
    calc_2 = False
    calc_3 = False
    calc_4 = False

    verify_2.config(bg="red", text="Ruído não removido")
    status_calc.config(bg="red", text="Calcular")

    if (bias.get() == 1) & (lamp.get() == 0):
        verify_1.config(text="Remover Bias")
    elif (bias.get() == 0) & (lamp.get() == 1):
        verify_1.config(text="Remover Lamp")
    elif (bias.get() == 0) & (lamp.get() == 0):
        verify_1.config(text="Não remover nada")
    else:
        verify_1.config(text="Remover Bias e Lamp")


def sigma_menu(sig):
    if sig == 1:
        sigma_button.configure(state=tk.NORMAL)
        repeat_op_sig.set(3)
    elif sig == 2:
        sigma_button.configure(state=tk.DISABLED)
        repeat_op_sig.set(0)
    check()


def reset():
    global reduced_status, calc_1, calc_2, calc_3, calc_4

    reduced_status = False
    calc_1 = False
    calc_2 = False
    calc_3 = False
    calc_4 = False

    bias.set(0)
    lamp.set(0)
    save_ndata.set(0)
    meth1_op.set(0)
    meth2_op.set(0)
    meth3_op.set(0)
    meth4_op.set(0)
    save_meths.set(False)
    plot1_op.set(False)
    save_plot1_op.set(False)
    scale_plot1_op.set(True)
    plot2_op.set(False)
    save_plot2_op.set(False)
    meth1_show.set(False)
    meth2_show.set(False)
    meth3_show.set(False)
    meth4_show.set(False)
    inten_op.set(100)
    sigma_op.set(1)
    repeat_op.set(1)
    repeat_op_sig.set(3)
    op_sig.set("Mediana dos Mínimos e Máximos")

    sigma_button.configure(state=tk.NORMAL)
    verify_1.config(bg='lightgray', text="Não remover nada")
    verify_2.config(bg='red', text="Ruído não removido")
    status_calc.config(text="Calcular", command=calc, bg="red")
    log.config(text="")


def remove():
    global reduced, reduced_status
    global calc_1, calc_2, calc_3, calc_4

    calc_1 = False
    calc_2 = False
    calc_3 = False
    calc_4 = False

    if repeat_op.get() == 0:
        log.config(text="Repetições deve ser maior que 0.")
    else:
        log.config(text="")

        dt, noi_bias, noi_lamp = input_data()
        reduced = noise_removal(dt, noi_bias, noi_lamp, red_bias=bias.get(), red_lamp=lamp.get())

        if sigma_op.get() == 1:
            reduced = sigma_1(reduced, noi_bias, noi_lamp)
        elif sigma_op.get() == 2:
            reduced = sigma_2(reduced)

        if not reduced:
            reduced_status = False
            verify_2.config(bg='red', text="Ruído não removido")
        else:
            if save_ndata.get():
                save_new_data(reduced)
            reduced_status = True
            verify_2.config(bg='lightgreen', text="Ruído removido")

        status_calc.config(bg="red", text="Calcular")


def calc():
    global reduced, results_1, results_3
    global calc_1, calc_2, calc_3, calc_4

    if reduced_status:
        if meth1_op.get() == 1:
            results_1 = method_1(reduced, save=save_meths.get())
            calc_1 = True
        if meth3_op.get() == 1:
            results_3 = method_3(reduced, save=save_meths.get())
            calc_3 = True

        status_calc.config(bg="lightgreen", text="Calcular")

        if meth1_op.get() + meth2_op.get() + meth3_op.get() + meth4_op.get() == 0:
            status_calc.config(bg="red", text="Nenhum método selecionado")
    else:
        status_calc.config(bg="red", text="Aperte o botão \"Remover ruído\"")


def calc_reset():
    global calc_1, calc_2, calc_3, calc_4

    calc_1 = False
    calc_2 = False
    calc_3 = False
    calc_4 = False
    status_calc.config(bg="red", text="Calcular")


def plot_all_menu():
    global reduced, reduced_status, results_1, results_3

    show1 = False
    show2 = False
    show3 = False
    show4 = False

    if meth1_show.get() and calc_1:
        show1 = True
    if meth2_show.get() and calc_2:
        show2 = True
    if meth3_show.get() and calc_3:
        show3 = True
    if meth4_show.get() and calc_4:
        show4 = True

    if reduced_status:
        plot_all(reduced, results_1, results_3, show=plot1_op.get(), save=save_plot1_op.get(),
                 show_meth_1=show1, show_meth_3=show3)
    else:
        log.config(text="Aperte o botão \"Remover ruído\"")


def plot_same_menu():
    global reduced, reduced_status, results_1, results_3

    show1 = False
    show2 = False
    show3 = False
    show4 = False

    if meth1_show.get() and calc_1:
        show1 = True
    if meth2_show.get() and calc_2:
        show2 = True
    if meth3_show.get() and calc_3:
        show3 = True
    if meth4_show.get() and calc_4:
        show4 = True

    if reduced_status:
        plot_same_intensity(reduced, results_1, results_3, show=plot2_op.get(), save=save_plot2_op.get(),
                            k=inten_op.get(), show_meth_1=show1, show_meth_3=show3)
    else:
        log.config(text="Aperte o botão \"Remover ruído\"")


def plot_planck_menu():
    global results_1, results_3
    global calc_1, calc_2, calc_3, calc_4

    log.config(text="")

    if meth_planck.get() == 1 and calc_1:
        planck_meth(results_1, save=save_planck.get(), show=plot_planck.get())
    elif meth_planck.get() == 3 and calc_3:
        planck_meth(results_3, save=save_planck.get(), show=plot_planck.get())
    else:
        log.config(text="O método selecionado não\nfoi calculado.")


def main():
    ######################################################################################
    # Remove Noise menu
    tk.Label(window, text="Selecione os ruídos que deseja remover e clique em \"Remover ruído\".").grid(row=0, column=0, columnspan=4, sticky=tk.W+tk.E+tk.N+tk.S)
    bias_button = tk.Checkbutton(window, text="Bias", variable=bias, onvalue=1, offvalue=0, command=check)
    bias_button.grid(row=1, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
    lamp_button = tk.Checkbutton(window, text="Lamp", variable=lamp, onvalue=1, offvalue=0, command=check)
    lamp_button.grid(row=1, column=1, sticky=tk.W + tk.E + tk.N + tk.S)
    verify_1.grid(row=1, column=2, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Button(window, text="Remover ruído", command=remove).grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
    save_button = tk.Checkbutton(window, text="Salvar", variable=save_ndata, onvalue=True, offvalue=False)
    save_button.grid(row=2, column=2, sticky=tk.W + tk.E + tk.N + tk.S)
    verify_2.grid(row=2, column=3, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="{}".format("-"*100)).grid(row=3, column=0, columnspan=4, sticky=tk.W + tk.E + tk.N + tk.S)
    ######################################################################################
    # Methods menu
    tk.Label(window, text="Selecione os métodos que deseja calcular:").grid(row=4, column=0, columnspan=4, sticky=tk.W + tk.E + tk.N + tk.S)
    meth1_button = tk.Checkbutton(window, text="1", variable=meth1_op, onvalue=1, offvalue=0, command=calc_reset)
    meth1_button.grid(row=5, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
    meth2_button = tk.Checkbutton(window, text="2", variable=meth2_op, onvalue=1, offvalue=0, command=calc_reset, state=tk.DISABLED)
    meth2_button.grid(row=5, column=1, sticky=tk.W + tk.E + tk.N + tk.S)
    meth3_button = tk.Checkbutton(window, text="3", variable=meth3_op, onvalue=1, offvalue=0, command=calc_reset)
    meth3_button.grid(row=5, column=2, sticky=tk.W + tk.E + tk.N + tk.S)
    meth4_button = tk.Checkbutton(window, text="4", variable=meth4_op, onvalue=1, offvalue=0, command=calc_reset, state=tk.DISABLED)
    meth4_button.grid(row=5, column=3, sticky=tk.W + tk.E + tk.N + tk.S)
    status_calc.grid(row=6, column=0, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S)
    save_meths_button = tk.Checkbutton(window, text="Salvar", variable=save_meths, onvalue=True, offvalue=False)
    save_meths_button.grid(row=6, column=3, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="{}".format("-" * 100)).grid(row=7, column=0, columnspan=4, sticky=tk.W + tk.E + tk.N + tk.S)
    ######################################################################################
    # Plots menu
    tk.Label(window, text="Plotar e salvar gráficos\nMostrar resultado dos métodos:").grid(row=8, column=0, columnspan=4, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Checkbutton(window, text="1", variable=meth1_show, onvalue=True, offvalue=False).grid(row=9, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Checkbutton(window, text="2", variable=meth2_show, onvalue=True, offvalue=False, state=tk.DISABLED).grid(row=9, column=1, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Checkbutton(window, text="3", variable=meth3_show, onvalue=True, offvalue=False).grid(row=9, column=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Checkbutton(window, text="4", variable=meth4_show, onvalue=True, offvalue=False, state=tk.DISABLED).grid(row=9, column=3, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="{}".format("- " * 50)).grid(row=10, column=0, columnspan=4, sticky=tk.W + tk.E + tk.N + tk.S)
    #######################################
    tk.Label(window, text="Gráficos de todas\nas frequências:").grid(row=11, column=0, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    plot1_button = tk.Checkbutton(window, text="Mostrar", variable=plot1_op, onvalue=True, offvalue=False)
    plot1_button.grid(row=11, column=1, sticky=tk.W + tk.E + tk.N + tk.S)
    save_plot1_button = tk.Checkbutton(window, text="Salvar", variable=save_plot1_op, onvalue=True, offvalue=False)
    save_plot1_button.grid(row=11, column=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Button(window, text="Gerar", command=plot_all_menu).grid(row=11, column=3, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="{}".format("- " * 50)).grid(row=12, column=0, columnspan=4, sticky=tk.W + tk.E + tk.N + tk.S)
    scale_plot1_button = tk.Checkbutton(window, text="Padronizar Escala", variable=scale_plot1_op, onvalue=True, offvalue=False)
    scale_plot1_button.grid(row=11, column=4, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    #######################################
    tk.Label(window, text="Gráfico de uma\nintensidade:").grid(row=13, column=0, columnspan=1, rowspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="Escolha uma intensidade:").grid(row=13, column=1, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.OptionMenu(window, inten_op, *[100, 80, 60, 40, 20]).grid(row=13, column=3, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    plot2_button = tk.Checkbutton(window, text="Mostrar", variable=plot2_op, onvalue=True, offvalue=False)
    plot2_button.grid(row=14, column=1, sticky=tk.W + tk.E + tk.N + tk.S)
    save_plot2_button = tk.Checkbutton(window, text="Salvar", variable=save_plot2_op, onvalue=True, offvalue=False)
    save_plot2_button.grid(row=14, column=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Button(window, text="Gerar", command=plot_same_menu).grid(row=14, column=3, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    ######################################################################################
    # Sigma menu
    tk.Label(window, text="Digite o número de repetições global:").grid(row=0, column=4, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Entry(window, width=2, textvariable=repeat_op).grid(row=0, column=6, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="Qual tipo de redução do sigma?").grid(row=1, column=4, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S)
    sigma_reduc_button.grid(row=2, column=4, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="Selecione o tipo de sigma:").grid(row=3, column=4, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.OptionMenu(window, sigma_op, *[1, 2], command=sigma_menu).grid(row=3, column=6, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="Digite o número de repetições sigma 1:").grid(row=4, column=4, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    sigma_button.grid(row=4, column=6, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    ######################################################################################
    # Log menu
    tk.Label(window, text="Log:").grid(row=5, column=4, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S)
    log.grid(row=6, column=4, columnspan=3, rowspan=5, sticky=tk.W + tk.E + tk.N + tk.S)
    ######################################################################################
    # planck Menu
    tk.Label(window, text="Gráfico dos V0s pelo Comprimento de Onda:").grid(row=12, column=4, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="Qual método calcular:").grid(row=13, column=4, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.OptionMenu(window, meth_planck, *[1, 3]).grid(row=13, column=6, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    plotplanck_button = tk.Checkbutton(window, text="Mostrar", variable=plot_planck, onvalue=True, offvalue=False)
    plotplanck_button.grid(row=14, column=4, sticky=tk.W + tk.E + tk.N + tk.S)
    save_planck_button = tk.Checkbutton(window, text="Salvar", variable=save_planck, onvalue=True, offvalue=False)
    save_planck_button.grid(row=14, column=5, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Button(window, text="Gerar", command=plot_planck_menu).grid(row=14, column=6, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    ######################################################################################
    # End menu
    tk.Button(window, text="Reset", command=reset).grid(row=15, column=0, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Button(window, text="Quit", command=window.quit).grid(row=15, column=2, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    window.mainloop()


results_1 = {}
results_3 = {}
reduced = {}
reduced_status = False
calc_1 = False
calc_2 = False
calc_3 = False
calc_4 = False

bias = tk.IntVar()
lamp = tk.IntVar()
save_ndata = tk.BooleanVar()
meth1_op = tk.IntVar()
meth2_op = tk.IntVar()
meth3_op = tk.IntVar()
meth4_op = tk.IntVar()
save_meths = tk.BooleanVar()
plot1_op = tk.BooleanVar()
save_plot1_op = tk.BooleanVar()
scale_plot1_op = tk.BooleanVar()
scale_plot1_op.set(True)
plot2_op = tk.BooleanVar()
save_plot2_op = tk.BooleanVar()
plot_planck = tk.BooleanVar()
save_planck = tk.BooleanVar()
meth_planck = tk.IntVar()
meth_planck.set(1)
meth1_show = tk.BooleanVar()
meth2_show = tk.BooleanVar()
meth3_show = tk.BooleanVar()
meth4_show = tk.BooleanVar()
inten_op = tk.IntVar()
inten_op.set(100)
repeat_op = tk.IntVar()
repeat_op.set(1)
sigma_op = tk.IntVar()
sigma_op.set(1)
op_sig = tk.StringVar()
op_sig.set("Mediana dos Mínimos e Máximos")
repeat_op_sig = tk.IntVar()
repeat_op_sig.set(3)

verify_1 = tk.Label(window, bg='lightgray', text="Não remover nada")
verify_2 = tk.Label(window, bg='red', text="Ruído não removido")
status_calc = tk.Button(window, text="Calcular", command=calc, bg="red")
sigma_button = tk.Entry(window, width=2, textvariable=repeat_op_sig)
sigma_reduc_button = tk.OptionMenu(window, op_sig, *["Mediana dos Mínimos e Máximos", "Mínimos e Máximos", "Desvio Padrão"],
                              command=sigma_menu)
log = tk.Label(window, text="", bg="white")

main()

