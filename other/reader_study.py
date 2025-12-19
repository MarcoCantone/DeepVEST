import pandas as pd
import numpy as np

answer1 = pd.read_csv(r"C:\Users\marco\Desktop\MRI\Reader Study\reader_study_breast-mri-sefe-mri_answers_1_1000_2025-04-14T10_46_45.238541+00_00.csv")
answer2 = pd.read_csv(r"C:\Users\marco\Desktop\MRI\Reader Study\reader_study_breast-mri-sefe-mri_answers_1001_2000_2025-04-14T10_32_03.809073+00_00.csv")
answer = pd.concat([answer1, answer2])

creators = list(answer["creator"].unique())

display_set_csv = pd.read_csv(r"C:\Users\marco\Desktop\Reader Study\reader_study_breast-mri-sefe-mri_display_sets_1_1000_2025-04-14T10_32_03.809073+00_00.csv")

cases = list(display_set_csv["api_url"].unique())

question_csv = pd.read_csv(r"C:\Users\marco\Desktop\Reader Study\reader_study_breast-mri-sefe-mri_questions_2025-04-14T10_32_03.809073+00_00.csv")
id_title_association = {}
for i in range(len(question_csv)):
    options = eval(question_csv.loc[i]["options"].replace("true", "True").replace("false", "False"))
    for option in options:
        id_title_association[option["id"]] = option["title"]
questions = question_csv["api_url"].unique()

res = {}
for creator in creators:
    res[creator] = {}
    for case in cases:
        res[creator][case] = {}
        for question in questions:
            res[creator][case][question] = answer[(answer["creator"] == creator) & (answer["display_set"].str.contains(case)) & (
                answer["question"].str.contains(question))]["answer"].item()

# # CREATE CSV
# print(f'Creator, Case, Question')
# for creator in creators:
#     for i, case in enumerate(cases):
#         for question in questions:
#             if ((question in questions[:2]) and (i % 2 == 0)) or ((question in questions[2:]) and (i % 2 == 1)):
#                 print(f'{creator}, {case}, {question}, {id_title_association[res[creator][case][question]]}')

z = []
for creator in creators:
    y = []
    for case in cases:
        x = []
        for question in questions:
            x.append(answer[(answer["creator"] == creator) & (answer["display_set"].str.contains(case)) & (
                answer["question"].str.contains(question))]["answer"].item())
        y.append(x)
    z.append(y)

res_arr = np.array(res)
res_arr = res_arr.transpose(0, 2, 1)

res = []
for i in range(5):
    question0 = [val for idx, val in enumerate(res_arr[i, 0]) if idx % 2 == 0]
    question1 = [val for idx, val in enumerate(res_arr[i, 1]) if idx % 2 == 0]
    question2 = [val for idx, val in enumerate(res_arr[i, 2]) if idx % 2 == 1]
    question3 = [val for idx, val in enumerate(res_arr[i, 3]) if idx % 2 == 1]
    res.append([question0, question1, question2, question3])

for i in range(5):
    for j in range(4):
        for k in range(30):
            res[i][j][k] = id_title_association[res[i][j][k]]

res_arr2 = np.array(res)