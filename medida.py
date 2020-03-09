################################################################
# Redução de Dados Física Experimental V - Efeito Fotoelétrico #
# Matheus J. Castro                                            #
# Version 6.0                                                  #
# Last Modification: 8 de Março de 2020                        #
################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk

window = tk.Tk()
width = 505
height = 430
posRight = int(window.winfo_screenwidth() / 2 - width / 2)
posDown = int(window.winfo_screenheight() / 2 - height / 2)
window.geometry("{}x{}+{}+{}".format(int(width), int(height), posRight, posDown))
window.title("Efeito Fotoelétrico por Matheus J. Castro")

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


def plot_all(data, meth_1, meth_3, show=True, save=False, show_meth_1=False, show_meth_3=False):
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
    if show:
        plt.show()
    else:
        plt.close()


def plot_same_intensity(data, meth_1, meth_3, k=100, show=True, save=False, show_meth_1=False, show_meth_3=False):
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
    if show:
        plt.show()
    else:
        plt.close()


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
    plot2_op.set(False)
    save_plot2_op.set(False)
    meth1_show.set(False)
    meth2_show.set(False)
    meth3_show.set(False)
    meth4_show.set(False)
    inten_op.set(100)

    verify_1.config(bg='lightgray', text="Não remover nada")
    verify_2.config(bg='red', text="Ruído não removido")
    status_calc.config(text="Calcular", command=calc, bg="red")


def remove():
    global reduced, reduced_status

    dt, noi_bias, noi_lamp = input_data()
    reduced = noise_removal(dt, noi_bias, noi_lamp, bias=bias.get(), lamp=lamp.get())

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
            calc_1 =True
        if meth3_op.get() == 1:
            results_3 = method_3(reduced, save=save_meths.get())
            calc_3 = True

        status_calc.config(bg="lightgreen", text="Calcular")

        if meth1_op.get() + meth2_op.get() + meth3_op.get() + meth4_op.get() == 0:
            status_calc.config(bg="red", text="Nenhum método selecionado")
    else:
        status_calc.config(bg="red", text="Aperte o \"botão Remover ruído\"")


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
        plot_same_intensity(reduced, results_1, results_3, show=plot2_op.get(), save=False, k=inten_op.get(), show_meth_1=show1, show_meth_3=show3)


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
    meth1_button = tk.Checkbutton(window, text="1", variable=meth1_op, onvalue=1, offvalue=0)
    meth1_button.grid(row=5, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
    meth2_button = tk.Checkbutton(window, text="2", variable=meth2_op, onvalue=1, offvalue=0, state=tk.DISABLED)
    meth2_button.grid(row=5, column=1, sticky=tk.W + tk.E + tk.N + tk.S)
    meth3_button = tk.Checkbutton(window, text="3", variable=meth3_op, onvalue=1, offvalue=0)
    meth3_button.grid(row=5, column=2, sticky=tk.W + tk.E + tk.N + tk.S)
    meth4_button = tk.Checkbutton(window, text="4", variable=meth4_op, onvalue=1, offvalue=0, state=tk.DISABLED)
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
    #######################################
    tk.Label(window, text="Gráfico de uma\nintensidade:").grid(row=13, column=0, columnspan=1, rowspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Label(window, text="Escolha uma intensidade:").grid(row=13, column=1, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.OptionMenu(window, inten_op, *[100, 80, 60, 40, 20]).grid(row=13, column=3, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
    plot1_button = tk.Checkbutton(window, text="Mostrar", variable=plot2_op, onvalue=True, offvalue=False)
    plot1_button.grid(row=14, column=1, sticky=tk.W + tk.E + tk.N + tk.S)
    save_plot1_button = tk.Checkbutton(window, text="Salvar", variable=save_plot2_op, onvalue=True, offvalue=False)
    save_plot1_button.grid(row=14, column=2, sticky=tk.W + tk.E + tk.N + tk.S)
    tk.Button(window, text="Gerar", command=plot_same_menu).grid(row=14, column=3, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)
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
plot2_op = tk.BooleanVar()
save_plot2_op = tk.BooleanVar()
meth1_show = tk.BooleanVar()
meth2_show = tk.BooleanVar()
meth3_show = tk.BooleanVar()
meth4_show = tk.BooleanVar()
inten_op = tk.IntVar()
inten_op.set(100)

verify_1 = tk.Label(window, bg='lightgray', text="Não remover nada")
verify_2 = tk.Label(window, bg='red', text="Ruído não removido")
status_calc = tk.Button(window, text="Calcular", command=calc, bg="red")

main()
