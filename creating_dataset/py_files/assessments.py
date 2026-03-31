#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


np.random.seed(42)


# In[3]:


students    = pd.read_csv("../datasets/students.csv")
enrollments = pd.read_csv("../datasets/enrollments.csv")
courses     = pd.read_csv("../datasets/courses.csv")


# In[4]:


difficulty_map = courses.set_index("course_id")["difficulty"].to_dict()
student_map = students.set_index('student_id').to_dict('index')


# In[5]:


assessments = [
    ("assignment", 1,  2),
    ("quiz",       1,  3),
    ("assignment", 2,  4),
    ("quiz",       2,  6),
    ("assignment", 3,  7),
    ("midterm",    1,  8),
    ("assignment", 4,  10),
    ("quiz",       3,  11),
    ("project",    1,  12),
    ("final",      1,  15),
]


# In[6]:


def generate_scores(student, student_id, course_id):
    ability = student['base_ability']
    effort = student['effort_level']
    difficulty = difficulty_map[course_id]

    trend = (effort - 0.5) * 0.15

    skip_prob = 0.02 + (1-effort) * 0.20

    scores = []
    for assess_type, assess_num, week in assessments:

        if assess_type == "assignment":
            if np.random.random() < skip_prob:
                scores.append({
                    "student_id":      student_id,
                    "course_id":       course_id,
                    "assessment_type": assess_type,
                    "assessment_num":  assess_num,
                    "score":           0.0,
                    "max_score":       100,
                    "week":            week,
                    "skipped":         1,
                })
                continue

        base = ability * 0.75 + effort * 0.60 + np.random.normal(0, 0.08) + 0.10

        time_effect = trend * (week /15)

        difficulty_penalty = difficulty * 0.15

        type_penalties = {
            "final":      0.08,
            "midterm":    0.05,
            "project":    0.03,
            "quiz":       0.02,
            "assignment": 0.00,
        }
        type_penalty = type_penalties[assess_type]

        raw = base + time_effect - difficulty_penalty - type_penalty
        score = round(np.clip(raw * 100, 0, 100), 1)

        scores.append({
            "student_id": student_id,
            "course_id": course_id,
            "assessment_type": assess_type,
            'assessment_num': assess_num,
            "score": score,
            "max_score": 100,
            "week" : week,
            "skipped": 0
        })

    return scores


# In[7]:


all_scores = []

for _, enrollment in enrollments.iterrows():
    student = student_map[enrollment['student_id']]
    course_id = enrollment['course_id']
    scores = generate_scores(student,enrollment['student_id'], course_id)
    all_scores.extend(scores)


# In[8]:


scores_df = pd.DataFrame(all_scores)
scores_df.to_csv("../datasets/assessment_scores.csv", index=False)


# In[9]:


print(f"✓ assessment_scores.csv: {len(scores_df)} rows")
print(f"\nAssessments per type:\n{scores_df.assessment_type.value_counts()}")
print(f"\nAverage score by type:")
print(scores_df.groupby("assessment_type")["score"].mean().round(1))
print(f"\nSkipped assignments: {scores_df['skipped'].sum()} ({scores_df['skipped'].mean():.2%})")
print(f"\nAverage score by week:")
print(scores_df.groupby("week")["score"].mean().round(1))


# In[ ]:




