#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


np.random.seed(42)


# In[3]:


courses_raw = [
    ("C001", "Intro to Programming",        1, "CS",   "hands_on"),
    ("C002", "Discrete Mathematics",        1, "MATH", "analytical"),
    ("C003", "Data Literacy",               1, "DS",   "visual"),
    ("C004", "Web Fundamentals",            1, "SE",   "hands_on"),
    ("C005", "Intro to Databases",          1, "IT",   "analytical"),
    ("C006", "Computer Systems",            1, "CS",   "analytical"),
    ("C007", "Probability and Stats",       1, "MATH", "analytical"),
    ("C008", "UX Basics",                   1, "SE",   "visual"),
    ("C009", "Intro to Networking",         1, "IT",   "visual"),
    ("C010", "Linear Algebra",              1, "MATH", "analytical"),

    ("C011", "Data Structures",             2, "CS",   "analytical"),
    ("C012", "Algorithms",                  2, "CS",   "analytical"),
    ("C013", "Statistical Inference",       2, "MATH", "analytical"),
    ("C014", "Exploratory Data Analysis",   2, "DS",   "visual"),
    ("C015", "Database Systems",            2, "IT",   "analytical"),
    ("C016", "Frontend Development",        2, "SE",   "hands_on"),
    ("C017", "Operating Systems",           2, "CS",   "analytical"),
    ("C018", "Network Fundamentals",        2, "IT",   "visual"),
    ("C019", "Object Oriented Programming", 2, "CS",   "hands_on"),
    ("C020", "Applied Statistics",          2, "MATH", "analytical"),
    ("C021", "UI/UX Design",                2, "SE",   "visual"),
    ("C022", "Systems Programming",         2, "CS",   "hands_on"),

    ("C023", "Machine Learning",            3, "DS",   "analytical"),
    ("C024", "Deep Learning",               3, "DS",   "analytical"),
    ("C025", "Software Engineering",        3, "SE",   "hands_on"),
    ("C026", "Distributed Systems",         3, "CS",   "analytical"),
    ("C027", "Data Pipelines",              3, "DS",   "hands_on"),
    ("C028", "Computer Vision",             3, "DS",   "visual"),
    ("C029", "NLP Fundamentals",            3, "DS",   "exploratory"),
    ("C030", "Cloud Architecture",          3, "IT",   "exploratory"),
    ("C031", "Cybersecurity Fundamentals",  3, "IT",   "analytical"),
    ("C032", "Mobile Development",          3, "SE",   "hands_on"),

    ("C033", "AI Systems Capstone",         4, "DS",   "exploratory"),
    ("C034", "Full Stack Capstone",         4, "SE",   "hands_on"),
    ("C035", "Data Engineering Capstone",   4, "DS",   "hands_on"),
    ("C036", "Systems Capstone",            4, "CS",   "analytical"),
    ("C037", "Cloud Capstone",              4, "IT",   "exploratory"),
    ("C038", "Cybersecurity Capstone",      4, "IT",   "analytical"),
    ("C039", "Research Methods Capstone",   4, "DS",   "exploratory"),
    ("C040", "Product Engineering Capstone",4, "SE",   "hands_on"),
]

courses = pd.DataFrame(courses_raw,
    columns=["course_id", "course_name", "tier", "department", "best_modality"])


# In[4]:


tier_difficulty_base = {1: 0.20, 2: 0.42, 3: 0.65, 4: 0.83}

difficulty = []

for tier in courses["tier"]:
    base = tier_difficulty_base[tier]
    noise = np.random.uniform(-0.05, 0.05)
    difficulty.append(round(np.clip(base + noise, 0, 1), 2))

courses['difficulty'] = difficulty

tier_passrate_base = {1: 0.95, 2: 0.85, 3: 0.75, 4:0.68}

pass_rate = []

for tier in courses['tier']:
    base = tier_passrate_base[tier]
    noise = np.random.uniform(-0.04, 0.04)
    pass_rate.append(round(np.clip(base + noise, 0.65, 0.97), 2))

courses['pass_rate'] = pass_rate

courses.to_csv("../datasets/courses.csv", index = False)


# In[5]:


print(f"✓ courses.csv: {len(courses)} rows")
print(f"\nCourses per tier:\n{courses.tier.value_counts().sort_index()}")
print(f"\nDifficulty by tier:")
print(courses.groupby("tier")["difficulty"].agg(["min", "mean", "max"]).round(2))
print(f"\nPass rate by tier:")
print(courses.groupby("tier")["pass_rate"].agg(["min", "mean", "max"]).round(2))
print(f"\nModality distribution:\n{courses.best_modality.value_counts()}")


# In[ ]:




