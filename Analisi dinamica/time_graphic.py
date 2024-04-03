import pandas as pd
import os
import re
import matplotlib.pyplot as plt

def parse_filename(filename):
    pattern = r"instance_(\d+)_freq_(\d+)_WT_(\d+)_type_(\d+)\.csv"
    match = re.search(pattern, filename)
    if match:
        return {
            "instance": int(match.group(1)),
            "frequency": int(match.group(2)),
            "WT": int(match.group(3)),
            "type": int(match.group(4))
        }
    else:
        return None

data = []
directory_path = './'

for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        params = parse_filename(filename)
        if params:
            filepath = os.path.join(directory_path, filename)
            df = pd.read_csv(filepath)
            total_execution_time = df["Execution Time"].sum()
            data.append({
                "instance": params["instance"],
                "frequency": params["frequency"],
                "WT": params["WT"],
                "total_execution_time": total_execution_time
            })

execution_df = pd.DataFrame(data)

total_time_by_instance = execution_df.groupby('instance')['total_execution_time'].sum()
total_hours = total_time_by_instance.sum() / 3600
ore_totali = int(total_hours)
minuti_totali = int((total_hours - ore_totali) * 60)
plt.figure(figsize=(15, 5))
plt.plot(total_time_by_instance.index, total_time_by_instance.values, marker='o')
plt.title('Tempo totale di esecuzione per istanza')
plt.xlabel('Istanza')
plt.ylabel('Tempo di esecuzione totale (s)')
plt.figtext(0.5, 0.01, f'Tempo di esecuzione totale: {ore_totali} ore e {int(minuti_totali)} minuti', ha='center')
plt.grid(True)
plt.show()

average_time_by_freq_wt = execution_df.groupby(['frequency', 'WT'])['total_execution_time'].mean().unstack()

average_time_by_freq_wt.index = average_time_by_freq_wt.index.astype(int)

plt.figure(figsize=(15, 5))
for wt in average_time_by_freq_wt.columns:
    filtered_df = average_time_by_freq_wt[average_time_by_freq_wt.index.isin([1, 2, 7, 14])]
    plt.plot(filtered_df.index, filtered_df[wt], marker='o', label=f'WT {wt}')
    
plt.xticks([1, 2, 7, 14])
plt.title('Tempo medio di esecuzione in base a Frequenza e WT')
plt.xlabel('Frequenza')
plt.ylabel('Tempo medio di esecuzione (s)')
plt.legend()
plt.grid(True)
plt.show()
