import pandas as pd
import matplotlib.pyplot as plt

# Carica i dati dal file CSV
data = pd.read_csv('./overall_simulation_results.csv')

# Calcola la media dei tempi di esecuzione, del numero di gruppi e delle lezioni individuali per ogni rapporto studenti-tutor
mean_data = data.groupby('student_to_tutor_ratio').mean().reset_index()

# Prepara i dati per il plotting
ratios = mean_data['student_to_tutor_ratio']
execution_times = mean_data['execution_time']
num_groups = mean_data['num_groups']
num_individual_tutoring = mean_data['num_individual_tutoring']

# Imposta la dimensione del grafico
plt.figure(figsize=(18, 6))

# Grafico a barre per i tempi di esecuzione medi
plt.subplot(1, 3, 1)
plt.bar(ratios, execution_times, color='skyblue')
plt.title('Media Tempi di Esecuzione')
plt.xlabel('Rapporto Studenti-Tutor')
plt.ylabel('Tempo (s)')
plt.xticks(rotation=45)

# Grafico a barre per il numero medio di gruppi
plt.subplot(1, 3, 2)
plt.bar(ratios, num_groups, color='lightgreen')
plt.title('Media Numero di Gruppi')
plt.xlabel('Rapporto Studenti-Tutor')
plt.ylabel('Numero di Gruppi')
plt.xticks(rotation=45)

# Grafico a barre per il numero medio di lezioni individuali
plt.subplot(1, 3, 3)
plt.bar(ratios, num_individual_tutoring, color='salmon')
plt.title('Media Lezioni Individuali')
plt.xlabel('Rapporto Studenti-Tutor')
plt.ylabel('Lezioni Individuali')
plt.xticks(rotation=45)

# Mostra il grafico
plt.tight_layout()
plt.show()
