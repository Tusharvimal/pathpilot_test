#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


np.random.seed(42)


# In[3]:


students = pd.read_csv('../datasets/students.csv')
courses = pd.read_csv('../datasets/courses.csv')
prereqs = pd.read_csv('../datasets/course_prerequisites.csv')

prereq_map = prereqs.groupby('course_id')['prereq_course_id'].apply(list).to_dict()


# In[4]:


tier1 = courses[courses.tier == 1]['course_id'].tolist()
tier2 = courses[courses.tier == 2]['course_id'].tolist()
tier3 = courses[courses.tier == 3]['course_id'].tolist()
tier4 = courses[courses.tier == 4]['course_id'].tolist()


# In[5]:


modality_map = courses.set_index('course_id')['best_modality'].to_dict()


# In[6]:


difficulty_map = courses.set_index('course_id')['difficulty'].to_dict()


# In[7]:


def generate_grade(student , course_id, completed_courses_grades):
    ability = student['base_ability']
    effort = student['effort_level']
    difficulty = difficulty_map[course_id]

    prereqs_needed = prereq_map.get(course_id, [])
    if prereqs_needed:
        prereq_grades   = [completed_courses_grades.get(p, ability * 3.5) for p in prereqs_needed]
        prereq_strength = np.mean(prereq_grades) / 4.0
    else:
        prereq_strength = ability * 0.85 + effort * 0.15

    course_modality = modality_map[course_id]
    student_modality = student['true_modality']
    modality_bonus = 0.35 if course_modality == student_modality else 0.0

    raw_score = (
        ability * 0.55 + 
        effort * 0.35 +
        prereq_strength * 0.05 + 
        modality_bonus * 0.15 + 
        np.random.normal(0, 0.03)
    )

    raw_score = raw_score * (1- difficulty * 0.15)
    grade = np.clip(raw_score * 4.0, 0.0, 4.0)
    return round(grade, 2)


# In[8]:


enrollments = []

for _, student in students.iterrows():
    completed = {}

    n_courses = np.random.choice([3,4,5], p = [0.3, 0.6, 0.1])

    tier1_shuffled = tier1.copy()
    np.random.shuffle(tier1_shuffled)
    sem1_courses = tier1_shuffled[:n_courses]
    sem2_courses = tier1_shuffled[n_courses: n_courses * 2]

    for sem, courses_this_sem in enumerate([sem1_courses, sem2_courses], start = 1):
        for course_id in courses_this_sem:
            grade = generate_grade(student, course_id, completed)
            passed = 1 if grade >= 1.5 else 0
            if grade >= 2.5:
                risk_class = 'Low'
            elif grade >= 1.5:
                risk_class = 'Medium'
            else:
                risk_class = 'High'
            completed[course_id] = grade
            enrollments.append({
                "student_id": student['student_id'],
                "course_id": course_id,
                "semester": sem, 
                "grade": grade,
                "passed": passed, 
                "modality_match": int(modality_map[course_id] == student['true_modality']),
                "risk_class": risk_class
            })

    tier2_shuffled = tier2.copy()
    np.random.shuffle(tier2_shuffled)
    sem3_courses = tier2_shuffled[:n_courses]
    sem4_courses = tier2_shuffled[n_courses:n_courses*2]

    for sem, courses_this_sem in enumerate([sem3_courses, sem4_courses], start=3):
        for course_id in courses_this_sem:
            grade = generate_grade(student, course_id, completed)
            passed = 1 if grade >= 1.5 else 0
            if grade >= 2.5:
                risk_class = 'Low'
            elif grade >= 1.5:
                risk_class = 'Medium'
            else:
                risk_class = 'High'
            completed[course_id] = grade
            enrollments.append({
                "student_id":     student["student_id"],
                "course_id":      course_id,
                "semester":       sem,
                "grade":          grade,
                "passed":         passed,
                "modality_match": int(modality_map[course_id] == student["true_modality"]),
                "risk_class": risk_class
            })

    tier3_shuffled = tier3.copy()
    np.random.shuffle(tier3_shuffled)
    sem5_courses = tier3_shuffled[:n_courses]

    for course_id in sem5_courses:
        grade = generate_grade(student, course_id, completed)
        passed = 1 if grade >= 1.5 else 0
        if grade >= 2.5:
                risk_class = 'Low'
        elif grade >= 1.5:
            risk_class = 'Medium'
        else:
            risk_class = 'High'
        completed[course_id] = grade
        enrollments.append({
            "student_id":     student["student_id"],
            "course_id":      course_id,
            "semester":       5,
            "grade":          grade,
            "passed":         passed,
            "modality_match": int(modality_map[course_id] == student["true_modality"]),
            "risk_class": risk_class
        })

    tier4_shuffled = tier4.copy()
    np.random.shuffle(tier4_shuffled)
    sem6_courses = tier4_shuffled[:2]

    for course_id in sem6_courses:
        grade = generate_grade(student, course_id, completed)
        passed = 1 if grade >= 1.5 else 0
        if grade >= 2.5:
                risk_class = 'Low'
        elif grade >= 1.5:
            risk_class = 'Medium'
        else:
            risk_class = 'High'
        completed[course_id] = grade
        enrollments.append({
            "student_id":     student["student_id"],
            "course_id":      course_id,
            "semester":       6,
            "grade":          grade,
            "passed":         passed,
            "modality_match": int(modality_map[course_id] == student["true_modality"]),
            "risk_class": risk_class
        })


# In[9]:


enrollments_df = pd.DataFrame(enrollments)
enrollments_df.to_csv("../datasets/enrollments.csv", index=False)


# In[10]:


print(f"✓ enrollments.csv: {len(enrollments_df)} rows")
print(f"\nEnrollments per semester:\n{enrollments_df.semester.value_counts().sort_index()}")
print(f"\nOverall pass rate: {enrollments_df.passed.mean():.2%}")
print(f"\nPass rate by semester:")
print(enrollments_df.groupby('semester')['passed'].mean().round(2))
print(f"\nAverage grade by semester:")
print(enrollments_df.groupby('semester')['grade'].mean().round(2))


# In[11]:


print(f"\nrisk_class distribution:")
print(enrollments_df['risk_class'].value_counts().sort_index())
print(f"\nAvg grade by risk_class:")
print(enrollments_df.groupby('risk_class')['grade'].mean().round(3))

