import random
import pandas as pd
import numpy as np
import ast
import time
from gurobipy import GRB, Model, quicksum

def generate_registration_dates(n, days):
    # Genera date di registrazione casuali uniformemente distribuite.
    return np.random.randint(0, days, size=n)

def generate_mentor_registration_dates(num_students, num_mentors, days, student_instance_type=1):
    """
    Genera date di registrazione per i mentori basate sulla frequenza specificata
    per il tipo di studente. Per il tipo 1, un mentore ogni due giorni. Per il tipo 4,
    due mentori al giorno in media.
    """
    if student_instance_type == 1:
        registration_dates = np.arange(0, days, 2)
        if len(registration_dates) < num_mentors:
            additional_dates = np.random.choice(registration_dates, size=num_mentors - len(registration_dates), replace=True)
            registration_dates = np.concatenate((registration_dates, additional_dates))
        else:
            registration_dates = registration_dates[:num_mentors]
    elif student_instance_type == 2:
        registration_dates = np.arange(0, days)
    elif student_instance_type == 3:
        registration_dates = []
        for day in range(days):
            if len(registration_dates) + 2 <= num_mentors:
                registration_dates.append(day)
                registration_dates.append(day)
            elif len(registration_dates) < num_mentors:
                registration_dates.append(day)
            else:
                break

        registration_dates = np.array(registration_dates)
        np.random.shuffle(registration_dates)
        registration_dates = np.sort(registration_dates[:num_mentors])
    elif student_instance_type == 4:
        registration_dates = np.random.randint(0, days, size=num_mentors)

    np.random.shuffle(registration_dates)
    return registration_dates

def generate_leaving_dates(registration_dates, min_days=7, mean_stay=14, std_dev=2):
    leaving_days = mean_stay + std_dev * np.random.randn(len(registration_dates))
    leaving_days = np.maximum(leaving_days, min_days)
    leaving_days = np.round(leaving_days)
    leaving_dates = registration_dates + leaving_days
    return leaving_dates.astype(int)

def generate_mentors_leaving_dates(registration_dates, mean_stay=21, min_days=14):
    """
    I mentori sono più pazienti e lasciano il programma dopo più tempo
    """
    leaving_dates = registration_dates + np.random.randint(min_days, mean_stay + 1, size=len(registration_dates))
    return leaving_dates

def update_leaving_dates(df, matched_ids, additional_days, simulation_end_day, id_column):
    for id_value in matched_ids:
        current_leaving_date = df.loc[df[id_column] == id_value, 'Leaving Date'].iloc[0]
        new_leaving_date = min(current_leaving_date + additional_days, simulation_end_day)
        df.loc[df[id_column] == id_value, 'Leaving Date'] = new_leaving_date
    return df

def update_time_capacity(df, mentor_hours_dict):
    for mentor_id, hours_allocated in mentor_hours_dict.items():
        current_time_capacity = df.loc[df['Mentor ID'] == mentor_id, 'Time Capacity'].iloc[0]
        new_time_capacity = max(current_time_capacity - hours_allocated, 0)
        df.loc[df['Mentor ID'] == mentor_id, 'Time Capacity'] = new_time_capacity
    return df

def create_data_frames(student_instance_type, days, instance_id, new_students_generation):
    if new_students_generation:
        if student_instance_type == 1:
            students_per_day = 1
            num_mentors = days // 2
        elif student_instance_type == 2:
            students_per_day = 2
            num_mentors = days
        elif student_instance_type == 3:
            students_per_day = 3
            num_mentors = round(days * 1.5)
        elif student_instance_type == 4:
            students_per_day = 4
            num_mentors = days * 2
        else:
            raise ValueError("Tipo di istanza studente non supportato")

        num_students = students_per_day * days
        student_registration_dates = generate_registration_dates(num_students, days)
        student_leaving_dates = generate_leaving_dates(student_registration_dates)
        
        mentor_registration_dates = generate_mentor_registration_dates(num_students, num_mentors, days, student_instance_type)
        mentor_leaving_dates = generate_mentors_leaving_dates(mentor_registration_dates)

        total_schools = int(num_students * 0.67)
        schools = ["School " + (chr(65 + (i % 26)) if i < 26 else chr(65 + (i % 26)) + str(i // 26)) for i in range(total_schools)]

        subjects = ["Maths", "Science", "English", "History", "Geography", "Art", "Music", "Physical Education", "Computer Science", "Economics", "Biology", "Chemistry", "Physics", "Literature", "Foreign Language"]

        subjects_by_year = {
            4: ["Maths", "Science", "English", "Geography", "Art"],
            5: ["Maths", "English", "History", "Physical Education", "Foreign Language"],
            6: ["Maths", "English", "History", "Physical Education", "Foreign Language"],
            7: ["Maths", "English", "History", "Physical Education", "Foreign Language"],
            8: ["Maths", "English", "History", "Physical Education", "Computer Science"],
            9: ["Maths", "English", "History", "Physical Education", "Literature"],
            10: ["Maths", "Science", "English", "History", "Physical Education"],
            11: ["Maths", "English", "History", "Physical Education", "Economics"],
            12: ["Maths", "History", "English", "Physical Education", "Computer Science"]
        }

        students_df = []
        mentors_df = []

        max_classes_per_school = 5
        class_letters = [chr(65 + i) for i in range(max_classes_per_school)]

        for i in range(num_mentors):
            
            social_disadvantage_preference = random.choices([0, 1, 3], weights=[1, 1, 1], k=1)[0]
            
            group = random.choices([True, False], weights=[54, 46], k=1)[0]

            time_range = random.choices(
                [(1, 3), (4, 6), (7, 10)], 
                weights=[40, 40, 20], 
                k=1
            )[0]
            time_capacity = random.randint(time_range[0], time_range[1])

            if time_capacity < 4:
                num_subjects = random.randint(1, 3)
            elif time_capacity <= 6:
                num_subjects = random.randint(1, 4)
            else:
                num_subjects = random.randint(1, 5)

            subjects_list = random.sample(subjects, num_subjects)
            
            max_group_capacities = [random.randint(2, 5) for _ in subjects_list]
            
            social_index = random.choices([0, 1, 3], weights=[50, 40, 10], k=1)[0]
            student_year_preference = random.choices([0, 1, 2, 'N'], weights=[5, 20, 15, 60], k=1)[0]
            social_preference = random.choices([0, 1, 3], weights=[1, 1, 1], k=1)[0]
            gpm = random.choices(["N", "W", "M", "S"], weights=[85, 5, 5, 5], k=1)[0]

            gpm_pm = 0 if gpm == "N" else 1

            if gpm == "N":
                gpm_sm = 1
            elif gpm == "W":
                gpm_sm = 3
            else:  # Per M e S
                gpm_sm = 0

            if gpm == "N":
                gpm_wm = 0
            elif gpm == "W":
                gpm_wm = 1.5
            elif gpm == "M":
                gpm_wm = 3
            else:  # Per S
                gpm_wm = 4.5

            mentors_df.append([i, subjects_list, max_group_capacities, social_disadvantage_preference,
                                group, time_capacity, social_index,
                                student_year_preference, social_preference, gpm, gpm_pm, gpm_sm, gpm_wm])
            
        from collections import Counter

        subjects_offered_by_mentors = [subject for mentor in mentors_df for subject in mentor[2]]
        subjects_distribution = Counter(subjects_offered_by_mentors)

        total_offers = sum(subjects_distribution.values())
        subjects_weights = {subject: count / total_offers for subject, count in subjects_distribution.items()}

        for i in range(num_students):
            student_school = random.choice(schools)

            student_class = random.choice(class_letters)
            
            social_disadvantage = random.choices([0, 1, 2, 3], weights=[65, 20, 10, 5], k=1)[0]
            
            accepts_group = random.choices([True, False], weights=[2, 1], k=1)[0]
            home_help = random.choice([0, 1, 2])
            equipment = 0
            year = random.randint(4, 12)
            
            num_subjects = random.choices([1, 2, 3, 4], weights=[50, 30, 10, 10], k=1)[0]
            
            available_subjects_for_year = subjects_by_year.get(year, [])

            available_weights = [subjects_weights.get(subject, 0) for subject in available_subjects_for_year]
            
            num_subjects = min(num_subjects, len(available_subjects_for_year))
            
            if sum(available_weights) > 0:
                total_weight = sum(available_weights)
                normalized_weights = [weight / total_weight for weight in available_weights]
                
                subjects_list = random.choices(available_subjects_for_year, weights=normalized_weights, k=num_subjects)
            else:
                subjects_list = random.sample(available_subjects_for_year, min(num_subjects, len(available_subjects_for_year)))
            
            time_list = random.choices([1,2,3,4], weights=[33, 33, 25, 9], k=num_subjects)
            
            grades_list = random.choices([0,1,2,3,4,5], k=num_subjects)
            
            single_parent = random.choices([False, True], weights=[81.7, 18.3], k=1)[0]
            
            num_parents = 1 if single_parent else 2
            
            siblings = random.choices([1, 2, 3, 4], weights=[69.2, 23.9, 5.2, 1.7], k=1)[0]
            
            not_enough_help_at_home = siblings / num_parents

            poisson_value = np.random.poisson(0.786)
            weak_student = min(poisson_value, 3)

            if year == 12:
                critical_year = 2
            elif year == 11:
                critical_year = 1
            else:
                critical_year = 0
                
            if year == 12:
                matriculation = 'advanced'
            elif year == 11:
                matriculation = np.random.choice(['N', 'basic', 'advanced'], p=[0.4, 0.4, 0.2])
            else:
                matriculation = 'N'
                
            students_df.append([i, student_school, student_class, subjects_list, time_list, grades_list, social_disadvantage,
                                accepts_group, home_help, equipment, year, not_enough_help_at_home, weak_student,
                                critical_year, matriculation])

        students_df = pd.DataFrame(students_df, columns=["Student ID", "School", "Class", "Subjects", "Time Required", "Grades", "Social Disadvantage",
                                                        "Accepts Group", "Home Help", "Equipment", "Year", "Not Enough Help at Home",
                                                        "Weak student", "Critical Year", "Matriculation"])
        mentors_df = pd.DataFrame(mentors_df, columns=[ "Mentor ID", "Subjects", "Max Group Capacity per Subject", "Social Disadvantage Preference", "Accepts Group", "Time Capacity", "Social Index",
            "Student Year Preference", "Social Preference", "GPM",
            "GPM_PM", "GPM_SM", "GPM_WM"])
        

        students_df['Registration Date'] = student_registration_dates
        students_df['Leaving Date'] = student_leaving_dates
        mentors_df['Registration Date'] = mentor_registration_dates
        mentors_df['Leaving Date'] = mentor_leaving_dates
        
        students_filename = f'students_data_type_{student_instance_type}_instance_{instance_id}.csv'
        mentors_filename = f'mentors_data_type_{student_instance_type}_instance_{instance_id}.csv'
        
        students_df.to_csv(students_filename, index=False)
        mentors_df.to_csv(mentors_filename, index=False)
        return students_df, mentors_df
    else:
        students_filename = f'students_data_type_{student_instance_type}_instance_{instance_id}.csv'
        mentors_filename = f'mentors_data_type_{student_instance_type}_instance_{instance_id}.csv'
        students_df = pd.read_csv(students_filename)
        mentors_df = pd.read_csv(mentors_filename)
        return students_df, mentors_df

def execute_model(students_df, mentors_df, current_day_matching_run, wt):

    students_updates = {}

    start_time = time.time()

    def string_to_list(string):
        if isinstance(string, list):
            return string
        try:
            return ast.literal_eval(string)
        except (ValueError, SyntaxError):
            return []
        
    students_df.loc[:, 'Subjects'] = students_df['Subjects'].apply(string_to_list)
    mentors_df.loc[:, 'Subjects'] = mentors_df['Subjects'].apply(string_to_list)
    mentors_df.loc[:, 'Max Group Capacity per Subject'] = mentors_df['Max Group Capacity per Subject'].apply(string_to_list)
    students_df.loc[:, 'Time Required'] = students_df['Time Required'].apply(string_to_list)
    students_df.loc[:, 'Grades'] = students_df['Grades'].apply(string_to_list)

    year_ranges = {
        0: list(range(1, 5)),
        1: list(range(5, 9)),
        2: list(range(9, 13)),
        'N': list(range(1, 13))
    }

    # 1. Studenti (A) e Tutor (B)
    A = set(students_df['Student ID'])
    B = set(mentors_df['Mentor ID'])

    students_accepting_group = set(students_df[students_df['Accepts Group']]['Student ID'])
    mentors_accepting_group = set(mentors_df[mentors_df['Accepts Group']]['Mentor ID'])

    # 2. Materie (S)
    subjects = ["Maths", "Science", "English", "History", "Geography", "Art", "Music", "Physical Education", "Computer Science", "Economics", "Biology", "Chemistry", "Physics", "Literature", "Foreign Language"]
    S = set(subjects)

    # 3. Materie richieste dagli studenti (S(ai))
    S_ai = {student_id: {(subject + " - Year " + str(year)) for subject in subjects}
            for student_id, subjects, year in zip(students_df['Student ID'], students_df['Subjects'], students_df['Year'])}

    # Materie offerte dai tutor (S(bj)
    S_bj = {}
    for index, row in mentors_df.iterrows():
        mentor_id = row['Mentor ID']
        mentor_subjects = row['Subjects']
        all_years = range(4, 13)
        S_bj[mentor_id] = {f"{subject} - Year {year}" for subject in mentor_subjects for year in all_years}  
        

    # 4. Attività (E)
    E = set()
    for student_id, student_subjects_years in S_ai.items():
        for subject_year in student_subjects_years:
            for mentor_id, mentor_subjects_years in S_bj.items():
                if subject_year in mentor_subjects_years:
                    E.add((student_id, mentor_id, subject_year))
                    
    E_group = {(student_id, mentor_id, subject) for student_id, mentor_id, subject in E if student_id in students_accepting_group and mentor_id in mentors_accepting_group}

    P = {}
    for mentor_id in mentors_accepting_group:
        for subject_year in S_bj[mentor_id]:
            for t in range(1, 6):
                P[(mentor_id, subject_year, t)] = None
                    
    def get_mentor_preferences(mentor_subjects_list):
        return mentor_subjects_list

    def get_student_preferences(student_subjects_list):
        return student_subjects_list

    PMj = {mentor_id: get_mentor_preferences(subjects_list) for mentor_id, subjects_list in zip(mentors_df['Mentor ID'], mentors_df['Subjects'])}
    PAi = {student_id: get_student_preferences(subjects_list) for student_id, subjects_list in zip(students_df['Student ID'], students_df['Subjects'])}

    c_jk = {}
    for index, mentor in mentors_df.iterrows():
        mentor_id = mentor['Mentor ID']
        subjects_offered = mentor['Subjects']
        capacities = mentor['Max Group Capacity per Subject']
        
        if isinstance(capacities, int):
            capacities = [capacities]
        
        capacity_dict = {subject: capacity for subject, capacity in zip(subjects_offered, capacities)}
        
        for subject_year in S_bj[mentor_id]:
            subject_name = subject_year.split(' - ')[0]
            if subject_name in subjects_offered:
                c_jk[(mentor_id, subject_year)] = capacity_dict[subject_name]


    # Dizionario per mappare ogni tutor alla sua capacità temporale settimanale
    Q = {mentor_id: mentors_df[mentors_df['Mentor ID'] == mentor_id]['Time Capacity'].iloc[0] for mentor_id in mentors_df['Mentor ID']}

    # Dizionario 'q' che mappa la tupla (student_id, subject_year) alle ore richieste
    q = {}
    for index, row in students_df.iterrows():
        student_id = row['Student ID']
        subjects = row['Subjects']
        time_required = row['Time Required']
        year = row['Year']
        
        if isinstance(time_required, int):
            time_required = [time_required]
        
        for subject, hours in zip(subjects, time_required):
            q[(student_id, f"{subject} - Year {year}")] = hours

            
    w_g = 0.7
    w_ew = 50
    w_m = 5

    # Dizionario per le preferenze di età dei tutor
    agepref_ij = {}
    for index, mentor in mentors_df.iterrows():
        mentor_id = mentor['Mentor ID']
        mentor_age_pref = mentor['Student Year Preference']
        if isinstance(mentor_age_pref, int):
            pass
        elif isinstance(mentor_age_pref, str) and mentor_age_pref.isdigit():
            mentor_age_pref = int(mentor_age_pref)
        else:
            mentor_age_pref = 'N'
            
            
        for student_id in A:
            student_year = students_df.loc[students_df['Student ID'] == student_id, 'Year'].values[0]
            student_year = int(student_year) if isinstance(student_year, str) and student_year.isdigit() else student_year
            
            if student_year in year_ranges.get(mentor_age_pref, []):
                agepref_ij[(student_id, mentor_id)] = 1
            else:
                agepref_ij[(student_id, mentor_id)] = 0

    rank_i = {student_id: {subject: idx for idx, subject in enumerate(subjects_list, start=1)} for student_id, subjects_list in PAi.items()}
    rank_j = {mentor_id: {subject: idx for idx, subject in enumerate(subjects_list, start=1)} for mentor_id, subjects_list in PMj.items()}

    E_first = set() # Include tutte le attività con rank_1 = 1
    w_ep = {}
    for e in E:
        student_id, mentor_id, subject_year = e
        subject = subject_year.split(' - ')[0]
        rank_i_value = rank_i[student_id][subject]
        rank_j_value = rank_j[mentor_id][subject]
        w_ep[e] = (6 - rank_i_value) + (6 - rank_j_value) + 3 * agepref_ij[(student_id, mentor_id)]
        if rank_i[student_id].get(subject) == 1:
            E_first.add(e)

    # gr(ai, sk) --> Il voto dello studente ai nella materia sk
    gr = {}
    for index, row in students_df.iterrows():
        student_id = row['Student ID']
        subjects = row['Subjects']
        grades = row['Grades']
        
        if not isinstance(grades, list):
            grades = [grades]
        
        for subject, grade in zip(subjects, grades):
            gr[(student_id, f"{subject} - Year {row['Year']}")] = grade

    sc_ii = {}
    for i in students_accepting_group:
        for i_prime in students_accepting_group:
            if i != i_prime:
                common_subject_years = S_ai[i].intersection(S_ai[i_prime])
                if common_subject_years:
                    same_class = students_df.loc[students_df['Student ID'] == i, 'Class'].values[0] == students_df.loc[students_df['Student ID'] == i_prime, 'Class'].values[0]
                    same_school = students_df.loc[students_df['Student ID'] == i, 'School'].values[0] == students_df.loc[students_df['Student ID'] == i_prime, 'School'].values[0]
                    same_year = students_df.loc[students_df['Student ID'] == i, 'Year'].values[0] == students_df.loc[students_df['Student ID'] == i_prime, 'Year'].values[0]
                    sc_ii[(i, i_prime)] = 1 if same_class and same_school and same_year else 0
                else:
                    sc_ii[(i, i_prime)] = 0

                
    dr_iik = {}
    dg_iik = {}
    for i in students_accepting_group:
        for i_prime in students_accepting_group:
            if i != i_prime:
                common_subject_years = S_ai[i].intersection(S_ai[i_prime])
                for subject_year in common_subject_years:
                    dr_iik[(i, i_prime, subject_year)] = abs(q[(i, subject_year)] - q[(i_prime, subject_year)])
                    dg_iik[(i, i_prime, subject_year)] = abs(gr[(i, subject_year)] - gr[(i_prime, subject_year)])
                    
    SD_i = {row['Student ID']: row['Social Disadvantage'] for index, row in students_df.iterrows()}
    WS_i = {row['Student ID']: row['Weak student'] for index, row in students_df.iterrows()}
    CY_i = {row['Student ID']: row['Critical Year'] for index, row in students_df.iterrows()}
    NH_i = {row['Student ID']: row['Not Enough Help at Home'] for index, row in students_df.iterrows()}
    help_i = {row['Student ID']: row['Home Help'] for index, row in students_df.iterrows()}

    DM_j = {row['Mentor ID']: row['Social Index'] for index, row in mentors_df.iterrows()}
    GPM_PM_j = {row['Mentor ID']: row['GPM_PM'] for index, row in mentors_df.iterrows()}
    GPM_SM_j = {row['Mentor ID']: row['GPM_SM'] for index, row in mentors_df.iterrows()}
    GPM_WM_j = {row['Mentor ID']: row['GPM_WM'] for index, row in mentors_df.iterrows()}

    # (15)
    w_et = {}
    for e in E:
        if e in E_first:
            student_id, mentor_id, subject_year = e
            student_registration_date = students_df.loc[students_df['Student ID'] == student_id, 'Registration Date'].values[0]
            mentor_registration_date = mentors_df.loc[mentors_df['Mentor ID'] == mentor_id, 'Registration Date'].values[0]
            w_et[e] = wt * ((current_day_matching_run - student_registration_date) + (current_day_matching_run - mentor_registration_date))
        else:
            w_et[e] = 0

    w_es = {}
    w_e = {}
    w_eg = {}
    for e in E:
        student_id, mentor_id, subject_year = e
        subject, year = subject_year.rsplit(' - ', 1)
        w_es[e] = (3 * SD_i[student_id] * DM_j[mentor_id] + WS_i[student_id] * GPM_SM_j[mentor_id] + (1.5 - abs(gr[student_id,subject_year] - GPM_WM_j[mentor_id])) * GPM_PM_j[mentor_id] + 2 * CY_i[student_id] + (NH_i[student_id] + help_i[student_id]))  
        w_e[e] = w_ew + w_ep[e] + w_es[e] + w_et[e]
        if e in E_group:
            for t in range(1,6):
                w_eg[(e,t)] = w_e[e] * w_g 

    model = Model("Online_Voluenteers")

    y_e = model.addVars(E, vtype=GRB.BINARY, name="y_e")

    x_e = model.addVars(E, vtype=GRB.INTEGER, name="x_e")

    y_jkt = {}
    x_jkt = {}
    for j in B:
        for sk in S_bj[j]:
            for t in range(1, 6):
                y_jkt[j, sk, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{sk}_{t}")
                x_jkt[j, sk, t] = model.addVar(vtype=GRB.INTEGER, name=f"x_{j}_{sk}_{t}")

    y_et = {}
    x_et = {}
    for e in E_group:
        for t in range(1,6):
            y_et[e, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{e}_{t}")
            x_et[e, t] = model.addVar(vtype=GRB.INTEGER, name=f"x_{e}_{t}")
            
    beta_ik = {}
    for i in A:
        for k in S_ai[i]:
            beta_ik[i, k] = model.addVar(vtype=GRB.BINARY, name=f"beta_{i}_{k}")
            
    M = 5 
    gamma_i = model.addVars(A, vtype=GRB.BINARY, name="gamma_i")

    z_iip = {}
    for i in students_accepting_group:
        for i_prime in students_accepting_group:
            if i != i_prime:
                for p in P:
                    z_iip[(i, i_prime, p)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{i_prime}_{p}")
                    
    m_ij = model.addVars(A, B, vtype=GRB.BINARY, name="m")
        
    # Vincolo (1)
    model.addConstrs((x_e[e] >= y_e[e] for e in E), "limite_inferiore")
    model.addConstrs((x_e[e] <= 3 * y_e[e] for e in E), "limite_superiore")

    # Vincolo (2) e (3)
    for j in B:
        for sk in S_bj[j]:
            activities_for_mentor_subject = [e for e in E_group if e[1] == j and e[2] == sk]
            for t in range(1, 6):
                model.addConstr(2 * y_jkt[j, sk, t] <= x_jkt[j, sk, t], name=f"min_hours_{j}_{sk}_{t}")
                model.addConstr(x_jkt[j, sk, t] <= 3 * y_jkt[j, sk, t], name=f"max_hours_{j}_{sk}_{t}")
                # Vincolo (3)
                sum_y_et = quicksum(y_et[e,t] for e in activities_for_mentor_subject)
                model.addConstr(2 * y_jkt[j, sk, t] <= sum_y_et, name=f"min_group_activities_{j}_{sk}_{t}")
                model.addConstr(sum_y_et <= c_jk[(j, sk)] * y_jkt[j, sk, t], name=f"max_group_activities_{j}_{sk}_{t}")
                    
    # Vincolo (4)
    for i in A:
        for sk in S_ai[i]:
            activities_for_student_subject = [e for e in E if e[0] == i and e[2] == sk]
            sum_y_e = quicksum(y_e[e] for e in activities_for_student_subject)
            sum_y_et = quicksum(y_et[e, t] if e in E_group else 0 for e in activities_for_student_subject for t in range(1, 6))
            model.addConstr(beta_ik[i, sk] == sum_y_e + sum_y_et, name=f"beta_constraint_{i}_{sk}")
            
    # Vincolo (5) 
    for i in A:
        sum_beta_ik = quicksum(beta_ik[i, sk] for sk in S_ai[i])
        model.addConstr(gamma_i[i] <= sum_beta_ik, name=f"gamma_min_{i}")
        model.addConstr(sum_beta_ik <= M * gamma_i[i], name=f"gamma_max_{i}")
        
    # Vincolo (6)
    for j in B:
        total_hours_per_mentor = quicksum(
            (quicksum(x_e[e] for e in E if e[1] == j and e[2] == sk) +
            quicksum(x_jkt[j, sk, t] for t in range(1, 6)))
            for sk in S_bj[j]
        )
        model.addConstr(total_hours_per_mentor <= Q[j], name=f"capacity_constraint_{j}")
            
    # Vincolo (7), (8) e (9)
    for e in E:
        ai, bj, sk = e
        # Vincolo (7)
        model.addConstr(x_e[e] <= q[(ai, sk)], name=f"student_request_constraint_{e}")
        if e in E_group:
            for t in range(1, 6):
                # Vincolo (8)
                model.addConstr(x_et[e, t] <= q[(ai, sk)], name=f"group_request_constraint_{e}_{t}")
                # Vincolo (9)
                model.addConstr(x_et[e, t] <= x_jkt[bj, sk, t], name=f"group_mentor_hours_constraint_{e}_{t}")
            
    # Vincoli (10), (11) e (12)
    for p in P:
        bj, sk, t = p
        for ai in students_accepting_group:
            for ai_prime in students_accepting_group:
                if ai != ai_prime:
                    e = (ai, bj, sk)
                    e_prime = (ai_prime, bj, sk)
                    if e in E_group and e_prime in E_group:
                        # Vincolo (10)
                        model.addConstr(z_iip[(ai, ai_prime, p)] >= y_et[e, t] + y_et[e_prime, t] - 1,
                                        name=f"z_constraint_{ai}_{ai_prime}_{bj}_{sk}_{t}")
                        # Vincolo (11)
                        model.addConstr(z_iip[(ai, ai_prime, p)] <= y_et[e, t],
                                        name=f"z_constraint_11_{ai}_{ai_prime}_{p}")
                        # Vincolo (12)
                        model.addConstr(z_iip[(ai, ai_prime, p)] <= y_et[e_prime, t],
                                        name=f"z_constraint_12_{ai}_{ai_prime}_{p}")

    z_iik = model.addVars(
        ((i, i_prime, sk) for i in students_accepting_group for i_prime in students_accepting_group if i != i_prime for sk in S_ai[i].intersection(S_ai[i_prime])),
        vtype=GRB.BINARY, name="z_iik"
    )
    for i in students_accepting_group:
        for i_prime in students_accepting_group:
            if i != i_prime:
                for sk in S_ai[i].intersection(S_ai[i_prime]):
                    model.addConstr(
                        z_iik[i, i_prime, sk] == quicksum(z_iip[i, i_prime, p] for p in P if p[1] == sk),
                        name=f"def_z_iik_{i}_{i_prime}_{sk}"
                    )
                    
    gc_z = quicksum((10 * sc_ii[i, i_prime] - dg_iik[i, i_prime, sk] - dr_iik[i, i_prime, sk]) * z_iik[i, i_prime, sk]
                for i in students_accepting_group for i_prime in students_accepting_group if i != i_prime
                for sk in S_ai[i].intersection(S_ai[i_prime]))

    # Vincolo (13)
    for i in A:
        for j in B:
            activities_ij = [e for e in E if e[0] == i and e[1] == j]
            model.addConstr(m_ij[i, j] <= quicksum(y_e[e] for e in activities_ij), name=f"m_constraint_upper_{i}_{j}")
            model.addConstr(quicksum(y_e[e] for e in activities_ij) <= 5 * m_ij[i, j], name=f"m_constraint_lower_{i}_{j}")

    mp_y = w_m * quicksum(m_ij[i, j] for i in A for j in B)

    sum_individual_mentoring = quicksum(w_e[e] * x_e[e] for e in E)

    sum_group_mentoring = quicksum(w_eg[e, t] * x_et[e, t] for e in E_group for t in range(1, 6))

    final_objective = sum_individual_mentoring + sum_group_mentoring + gc_z - mp_y

    model.setObjective(final_objective, GRB.MAXIMIZE)

    model.optimize()

    # Calcola il tempo di esecuzione
    execution_time = time.time() - start_time

    if model.status == GRB.Status.OPTIMAL:
        print('Valore ottimale della funzione obiettivo:', model.objVal)
    else:
        print('Soluzione ottimale non trovata. Stato del modello:', model.status)

    if model.status == GRB.Status.OPTIMAL:
        group_details = {}

        for (j, sk, t) in P:
            if y_jkt[j, sk, t].X > 0.5:
                group_students = []
                for e in E_group:
                    ai, bj, subject = e
                    if bj == j and subject == sk and y_et[(e, t)].X > 0.5:
                        group_students.append(ai)
                group_details[(j, sk, t)] = {
                    'Mentor': j,
                    'Subject': sk,
                    'Group Number': t,
                    'Students': group_students,
                    'Total Hours': x_jkt[j, sk, t].X
                }

        individual_activities = {}
        for e in E:
            if y_e[e].X > 0.5:
                activity_key = e
                activity_hours = x_e[e].X
                individual_activities[activity_key] = {
                    'Student': e[0],
                    'Mentor': e[1],
                    'Subject': e[2],
                    'Hours': activity_hours
                }
        matched_student_ids = set()
        matched_mentor_ids = set()
        mentor_hours_dict = {}
        for activity, details in individual_activities.items():
            matched_student_ids.add(details['Student'])
            matched_mentor_ids.add(details['Mentor'])
            mentor_hours_dict[details['Mentor']] = mentor_hours_dict.get(details['Mentor'], 0) + details['Hours']

        for group, details in group_details.items():
            matched_mentor_ids.add(details['Mentor'])
            mentor_hours_dict[details['Mentor']] = mentor_hours_dict.get(details['Mentor'], 0) + details['Total Hours']
            for student in details['Students']:
                matched_student_ids.add(student)

        volume = sum(x_e[e].X for e in E) + sum(w_g * x_et[e, t].X for e in E_group for t in range(1, 6))

        social_points_individual = sum(w_es[e] * y_e[e].X for e in E if y_e[e].X > 0.5)
        social_points_group = sum(w_es[e] * y_et[e, t].X for e in E_group for t in range(1, 6) if y_et[e, t].X > 0.5)
        social_points = social_points_individual + social_points_group

        preference_points_individual = sum(w_ep[e] * y_e[e].X for e in E if y_e[e].X > 0.5)
        preference_points_group = sum(w_ep[e] * y_et[e, t].X for e in E_group for t in range(1, 6) if y_et[e, t].X > 0.5)
        preference_points = preference_points_individual + preference_points_group

        students_involved = sum(gamma_i[i].X for i in A)

        for student_id in matched_student_ids:
            student_index = students_df.index[students_df['Student ID'] == student_id].tolist()[0]

            subjects_matched = set()
            for activity_key, details in individual_activities.items():
                if details['Student'] == student_id:
                    subjects_matched.add(details['Subject'].split(' - ')[0])

            for group_key, details in group_details.items():
                if student_id in details['Students']:
                    subjects_matched.add(details['Subject'].split(' - ')[0])

            subjects_list = students_df.at[student_index, 'Subjects']
            grades_list = students_df.at[student_index, 'Grades']
            time_list = students_df.at[student_index, 'Time Required']

            updated_subjects = [subject for subject in subjects_list if subject not in subjects_matched]
            updated_grades = [grades_list[i] for i, subject in enumerate(subjects_list) if subject not in subjects_matched]
            updated_times = [time_list[i] for i, subject in enumerate(subjects_list) if subject not in subjects_matched]

            students_df.at[student_index, 'Subjects'] = updated_subjects
            students_df.at[student_index, 'Grades'] = updated_grades
            students_df.at[student_index, 'Time Required'] = updated_times

            for student_id in matched_student_ids:
                student_index = students_df.index[students_df['Student ID'] == student_id].tolist()[0]
                student_updates = {
                    'Subjects': students_df.at[student_index, 'Subjects'],
                    'Grades': students_df.at[student_index, 'Grades'],
                    'Time Required': students_df.at[student_index, 'Time Required']
                }
                students_updates[student_id] = student_updates

        model.dispose() 
        return matched_student_ids, matched_mentor_ids, mentor_hours_dict, individual_activities, group_details, volume, social_points, preference_points, execution_time, students_updates, students_involved
    else:
        return set(), set(), {}, {}, {}, 0, 0, 0, 0, {}, 0

def print_matching_details(day, matched_student_ids, matched_mentor_ids, individual_activities, group_details):
    print(f"Dettagli abbinamenti per il giorno {day}:")
    print("Abbinamenti individuali:")
    for key, value in individual_activities.items():
        if value['Student'] in matched_student_ids:
            print(f"Studente: {value['Student']}, Tutor: {value['Mentor']}, Materia: {value['Subject']}, Ore: {value['Hours']}")
    print("Abbinamenti di gruppo:")
    for key, value in group_details.items():
        if value['Mentor'] in matched_mentor_ids:
            print(f"Tutor: {value['Mentor']}, Materia: {value['Subject']}, Numero Gruppo: {value['Group Number']}, Studenti: {value['Students']}, Ore Totali: {value['Total Hours']}")

def simulate_matching_over_time(students_df, mentors_df, start_day, end_day, matching_frequency, student_instance_type, simulation_duration, wt, instance_id):
    students_df = students_df.copy()
    mentors_df = mentors_df.copy()
    matching_results = []

    details_to_print = {}
    
    for day in range(start_day, end_day, matching_frequency):
        available_students = students_df[(students_df['Registration Date'] <= day) & (students_df['Leaving Date'] >= day)]
        available_mentors = mentors_df[(mentors_df['Registration Date'] <= day) & (mentors_df['Leaving Date'] >= day)]

        total_students_available = available_students.shape[0]
        total_mentors_available = available_mentors.shape[0]

        if not available_students.empty and not available_mentors.empty:
            matched_student_ids, matched_mentor_ids, mentor_hours_dict, individual_activities, group_details, volume, social_points, preference_points, execution_time, students_updates, students_involved = execute_model(available_students, available_mentors, day, wt)

            for student_id, updates in students_updates.items():
                student_index = students_df.index[students_df['Student ID'] == student_id].tolist()[0]
                students_df.at[student_index, 'Subjects'] = updates['Subjects']
                students_df.at[student_index, 'Grades'] = updates['Grades']
                students_df.at[student_index, 'Time Required'] = updates['Time Required']

            students_df = update_leaving_dates(students_df, matched_student_ids, additional_days=7, simulation_end_day=simulation_duration, id_column='Student ID')

            mentors_df = update_leaving_dates(mentors_df, matched_mentor_ids, additional_days=14, simulation_end_day=simulation_duration, id_column='Mentor ID')

            mentors_df = update_time_capacity(mentors_df, mentor_hours_dict)

            matching_results.append({
                'Day': day,
                'Available Students': total_students_available,
                'Available Mentors': total_mentors_available,
                'Individual Activities': len(individual_activities),
                'Group Activities': len(group_details),
                'Total Hours Allocated': sum(mentor_hours_dict.values()),
                'Volume': volume,
                'Social Points': social_points,
                'Preference Points': preference_points,
                'Students Involved': students_involved,
                'Execution Time': execution_time
            })
            print(f"Matching eseguito per il giorno {day}")
        else:
            print(f"Nessun matching disponibile per il giorno {day}")

    results_df = pd.DataFrame(matching_results)
    filename = f'matching_results_instance_{instance_id}_freq_{matching_frequency}_WT_{wt}_type_{student_instance_type}.csv'
    results_df.to_csv(filename, index=False)

def simulate_multiple_instances(instance_count, student_types, simulation_duration, matching_frequencies, wt_values):
    for student_type in student_types:
        for instance_id in range(1, instance_count + 1):
            students_df, mentors_df = create_data_frames(student_type, simulation_duration, instance_id, new_students_generation=False)
            for matching_frequency in matching_frequencies:
                for wt in wt_values:
                    simulate_matching_over_time(students_df, mentors_df, 0, simulation_duration, matching_frequency, student_type, simulation_duration, wt, instance_id)

simulation_duration = 300
simulate_multiple_instances(100, [1], simulation_duration, [1, 2, 7, 14], [0, 1, 2, 10])