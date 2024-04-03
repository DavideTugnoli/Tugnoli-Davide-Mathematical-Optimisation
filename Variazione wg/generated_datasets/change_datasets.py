import pandas as pd

def convert_accepts_group_to_true(file_name):
    try:
        df = pd.read_csv(file_name)
    
        if "Accepts Group" in df.columns:
            df["Accepts Group"] = True
            
            df.to_csv(file_name, index=False)
            print(f"File '{file_name}' aggiornato con successo a True.")
        else:
            print(f"La colonna 'Accepts Group' non è stata trovata in '{file_name}'.")
    except FileNotFoundError:
        print(f"File '{file_name}' non trovato.")

base_names = ["mentors_data_", "students_data_"]
for base_name in base_names:
    for i in range(1, 11):
        file_name = f"{base_name}{i}.csv"
        convert_accepts_group_to_true(file_name)
