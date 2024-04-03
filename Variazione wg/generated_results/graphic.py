import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def graphic_1():
    data_mixed = pd.read_csv('./simulation_results_mixed_t_f.csv')
    data_all_false = pd.read_csv('./simulation_results_all_false.csv')
    data_all_true = pd.read_csv('./simulation_results_tutti_t.csv')

    avg_execution_time_mixed = data_mixed.groupby('wg')['execution_time'].mean()
    avg_execution_time_all_false = data_all_false.groupby('wg')['execution_time'].mean()
    avg_execution_time_all_true = data_all_true.groupby('wg')['execution_time'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_execution_time_mixed, label='Distribuzione Mista', marker='o')
    plt.plot(avg_execution_time_all_false, label='Nessuno Accetta Gruppi', marker='s')
    plt.plot(avg_execution_time_all_true, label='Tutti Accettano Gruppi', marker='^')

    plt.title('Media del Tempo di Esecuzione al Variare del Peso di Gruppo wg (80 Studenti, 40 Tutor, 10 Dataset)')
    plt.xlabel('Peso di Gruppo wg')
    plt.ylabel('Tempo di Esecuzione Medio (s)')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0.5, 1.1, 0.1))
    plt.show()

def graphic_2():
    data_mixed = pd.read_csv('./simulation_results_mixed_t_f.csv')
    data_all_false = pd.read_csv('./simulation_results_all_false.csv')
    data_all_true = pd.read_csv('./simulation_results_tutti_t.csv')

    avg_num_groups_mixed = data_mixed.groupby('wg')['num_groups'].mean()
    avg_num_individual_tutoring_mixed = data_mixed.groupby('wg')['num_individual_tutoring'].mean()

    avg_num_groups_all_false = data_all_false.groupby('wg')['num_groups'].mean()
    avg_num_individual_tutoring_all_false = data_all_false.groupby('wg')['num_individual_tutoring'].mean()

    avg_num_groups_all_true = data_all_true.groupby('wg')['num_groups'].mean()
    avg_num_individual_tutoring_all_true = data_all_true.groupby('wg')['num_individual_tutoring'].mean()

    plt.figure(figsize=(14, 6))

    # Grafico per il numero medio di gruppi
    plt.subplot(1, 2, 1)
    plt.plot(avg_num_groups_mixed, label='Distribuzione Mista', marker='o')
    plt.plot(avg_num_groups_all_false, label='Nessuno Accetta Gruppi', marker='s')
    plt.plot(avg_num_groups_all_true, label='Tutti Accettano Gruppi', marker='^')
    plt.title('Numero Medio di Gruppi al Variare del Peso di Gruppo wg')
    plt.xlabel('Peso di Gruppo wg')
    plt.ylabel('Numero Medio di Gruppi')
    plt.grid(True)
    plt.xticks(np.arange(0.5, 1.1, 0.1))

    # Grafico per il numero medio di tutoraggi individuali
    plt.subplot(1, 2, 2)
    plt.plot(avg_num_individual_tutoring_mixed, label='Distribuzione Mista', marker='o')
    plt.plot(avg_num_individual_tutoring_all_false, label='Nessuno Accetta Gruppi', marker='s')
    plt.plot(avg_num_individual_tutoring_all_true, label='Tutti Accettano Gruppi', marker='^')
    plt.title('Numero Medio di Lezioni Individuali al Variare del Peso di Gruppo wg')
    plt.xlabel('Peso di Gruppo wg')
    plt.ylabel('Numero Medio di Lezioni Individuali')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.92))
    plt.grid(True)
    plt.xticks(np.arange(0.5, 1.1, 0.1))

    plt.tight_layout()
    plt.show()

graphic_1()
graphic_2()