import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

path = './'
all_files = glob.glob(path + "matching_results_instance_*_freq_*_WT_*_type_1.csv")

aggregated_data = []

for filename in all_files:
    df = pd.read_csv(filename)
    
    params = filename.split('_')
    freq = int(params[params.index('freq') + 1])
    wt = int(params[params.index('WT') + 1])
    
    totals = df[['Individual Activities', 'Group Activities', 'Volume', 'Social Points', 'Students Involved', 'Preference Points']].sum().to_dict()
    
    aggregated_data.append({
        'Frequency': freq,
        'Wt': wt,
        **totals
    })

aggregated_df = pd.DataFrame(aggregated_data)

sns.set_style("whitegrid")

padding_values = {}
for metric in ['Individual Activities', 'Group Activities', 'Volume', 'Social Points', 'Students Involved', 'Preference Points']:
    max_value = aggregated_df[metric].max()
    min_value = aggregated_df[metric].min()
    padding = (max_value - min_value) * 0.1
    padding_values[metric] = (min_value - padding, max_value + padding)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=aggregated_df, x='Frequency', y=metric, hue='Wt', palette="Set2")

    ax.set_ylim(*padding_values[metric])

    plt.title(f'Boxplot delle somme di {metric} per Frequenza e WT')
    plt.xlabel('Frequenza')
    plt.ylabel(f'Totale {metric}')

    plt.legend(title='WT')
    plt.tight_layout()
    plt.show()
