import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import time
from collections import Counter
from gurobipy import GRB, Model, quicksum

def execute_data_generation_code():

    num_students = 80
    num_mentors = 40

    initial_time_capacity = 10
    capacity_step = -1
    num_iterations = 10

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

    for iteration in range(1, num_iterations + 1):
        current_time_capacity = initial_time_capacity + (capacity_step * (iteration - 1))
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
            time_capacity = current_time_capacity

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
            
            # NHi index
            not_enough_help_at_home = siblings / num_parents
            poisson_value = np.random.poisson(0.786)
            weak_student = min(poisson_value, 3) # Weak student (WSi in the article)

            # Determine how critical the year is for the student # Critical Year
            if year == 12:
                critical_year = 2  # Last year of studies
            elif year == 11:
                critical_year = 1  # Penultimate year
            else:
                critical_year = 0  # Other years 
                
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
            
        students_df.to_csv(f'./generated_datasets/students_data_{iteration}.csv', index=False)
        mentors_df.to_csv(f'./generated_datasets/mentors_data_{iteration}.csv', index=False)

def run_model_for_all_datasets(num_iterations=10):
    try:
        risultati_iterazioni = []
        initial_time_capacity = 10
        for iteration in range(1, num_iterations + 1):
            current_time_capacity = initial_time_capacity - (iteration - 1)
            start_time = time.time()
            students_df = pd.read_csv(f'./generated_datasets/students_data_{iteration}.csv')
            mentors_df = pd.read_csv(f'./generated_datasets/mentors_data_{iteration}.csv')

            def string_to_list(string):
                try:
                    return ast.literal_eval(string)
                except ValueError:
                    return []

            students_df['Subjects'] = students_df['Subjects'].apply(string_to_list)
            mentors_df['Subjects'] = mentors_df['Subjects'].apply(string_to_list)
            mentors_df['Max Group Capacity per Subject'] = mentors_df['Max Group Capacity per Subject'].apply(string_to_list)
            students_df['Time Required'] = students_df['Time Required'].apply(string_to_list)
            students_df['Grades'] = students_df['Grades'].apply(string_to_list)

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

            w_ep = {}
            for e in E:
                student_id, mentor_id, subject_year = e
                subject = subject_year.split(' - ')[0]
                rank_i_value = rank_i[student_id][subject]
                rank_j_value = rank_j[mentor_id][subject]
                w_ep[e] = (6 - rank_i_value) + (6 - rank_j_value) + 3 * agepref_ij[(student_id, mentor_id)]

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

            w_es = {}
            w_e = {}
            w_eg = {}
            for e in E:
                student_id, mentor_id, subject_year = e
                subject, year = subject_year.rsplit(' - ', 1)
                w_es[e] = (3 * SD_i[student_id] * DM_j[mentor_id] + WS_i[student_id] * GPM_SM_j[mentor_id] + (1.5 - abs(gr[student_id,subject_year] - GPM_WM_j[mentor_id])) * GPM_PM_j[mentor_id] + 2 * CY_i[student_id] + (NH_i[student_id] + help_i[student_id]))  
                w_e[e] = w_ew + w_ep[e] + w_es[e]
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
                        # Vincolo (2)
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

            execution_time = time.time() - start_time

            # 3. Volume Objective
            for e in E_group:
                volume_objective = quicksum(x_e[e].X for e in E) + quicksum(w_g * x_et[e, t].X for e in E_group for t in range(1, 6))

            # 2. Number of pairs and groups
            for e in E:
                numbers_of_pairs = quicksum(y_e[e].X for e in E)
                numbers_of_groups = quicksum(w_g * y_et[e, t].X for e in E_group for t in range(1, 6))
                number_of_pairs_and_groups_objective = numbers_of_pairs + numbers_of_groups

            # 1. Number of students allocated
            for i in A:
                number_of_students_allocated_objective = quicksum(gamma_i[i].X for i in A)

            # Number of pairs
            for e in E:
                number_of_pairs_objective = quicksum(y_e[e].X for e in E)
            
            # Number of groups
            for e in E_group:
                number_of_groups_objective = quicksum(y_et[e, t].X for e in E_group for t in range(1, 6))

            # Number of paired mentoring hours
            for e in E:
                number_of_paired_mentoring_hours = quicksum(x_e[e].X for e in E)

            # Number of group mentoring hours
            for e in E_group:
                number_of_group_mentoring_hours = quicksum(x_et[e, t].X for e in E_group for t in range(1, 6))

            # Calcolo della capacità utilizzata per ogni tutor in questa iterazione
            capacita_usata_per_tutor = {tutor: sum(x_e[(i, tutor, sk)].X for i, _, sk in E if (i, tutor, sk) in x_e) + sum(x_jkt[(tutor, sk, t)].X for sk in S_bj[tutor] for t in range(1, 6)) for tutor in B}

            dati_iterazione = {
                "Iterazione": iteration,
                "Valore Obiettivo": model.ObjVal,
                "Tempo Esecuzione": execution_time,
                "Time Capacity": current_time_capacity,
                "Volume": volume_objective,
                "Number of Pairs and Groups": number_of_pairs_and_groups_objective,
                "Number of Students Allocated": number_of_students_allocated_objective,
                "Number of Pairs": number_of_pairs_objective,
                "Number of Groups": number_of_groups_objective,
                "Number of Paired Mentoring Hours": number_of_paired_mentoring_hours,
                "Number of Group Mentoring Hours": number_of_group_mentoring_hours,
                "Capacità Utilizzata per Tutor": capacita_usata_per_tutor
            }

            risultati_iterazioni.append(dati_iterazione)

            df_risultati = pd.DataFrame(risultati_iterazioni)

            df_risultati.to_csv("./generated_results/risultati_modello.csv", index=False)

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
            
                df_group_details = pd.DataFrame.from_records(list(group_details.values()))
                df_individual_activities = pd.DataFrame.from_records(list(individual_activities.values()))

                df_group_details.to_csv(f"./generated_allocations/group_details_iteration_{iteration}.csv", index=False)
                df_individual_activities.to_csv(f"./generated_allocations/individual_activities_iteration_{iteration}.csv", index=False)

                model.remove(model.getVars()) 
                model.remove(model.getConstrs())
                model.dispose()
            else:
                print('Soluzione ottimale non trovata. Stato del modello:', model.status)
    finally:
        model.dispose()

def generate_objective_graphs():
    df_risultati = pd.read_csv("./generated_results/risultati_modello.csv")

    time_capacity = df_risultati['Time Capacity']
    volume = df_risultati['Volume']
    number_of_students_allocated = df_risultati['Number of Students Allocated']
    number_of_pairs_and_groups = df_risultati['Number of Pairs and Groups']

    # Grafico del Volume di Attività di Mentoring
    plt.figure(figsize=(10, 6))
    plt.plot(time_capacity, volume, marker='o', linestyle='-', color='b', label='Volume di Mentoring')
    plt.title('Volume di Attività di Mentoring al Variare della Time Capacity')
    plt.xlabel('Time Capacity Settimanale dei Tutor')
    plt.ylabel('Volume di Mentoring (Ore)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Grafico del Numero di Studenti Allocati
    plt.figure(figsize=(10, 6))
    plt.plot(time_capacity, number_of_students_allocated, marker='^', linestyle='-', color='g', label='Studenti Allocati')
    plt.title('Numero di Studenti Allocati al Variare della Time Capacity')
    plt.xlabel('Time Capacity Settimanale dei Tutor')
    plt.ylabel('Numero di Studenti Allocati')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Grafico del Numero di Coppie e Gruppi Creati
    plt.figure(figsize=(10, 6))
    plt.plot(time_capacity, number_of_pairs_and_groups, marker='s', linestyle='-', color='r', label='Coppie e Gruppi')
    plt.title('Numero di Coppie e Gruppi al Variare della Time Capacity')
    plt.xlabel('Time Capacity Settimanale dei Tutor')
    plt.ylabel('Numero di Coppie e Gruppi')
    plt.grid(True)
    plt.legend()
    plt.show()

#execute_data_generation_code()
#run_model_for_all_datasets()
generate_objective_graphs()



