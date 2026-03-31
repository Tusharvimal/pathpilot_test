#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


np.random.seed(42)


# In[3]:


prereqs = [
    # ── TIER 2 ────────────────────────────────────────────────────────────────
    ("C011", "C001"),           # Data Structures → Intro to Programming
    ("C012", "C011"),           # Algorithms → Data Structures
    ("C012", "C002"),           # Algorithms → Discrete Math
    ("C013", "C007"),           # Statistical Inference → Probability
    ("C013", "C010"),           # Statistical Inference → Linear Algebra
    ("C014", "C003"),           # EDA → Data Literacy
    ("C014", "C007"),           # EDA → Probability
    ("C015", "C005"),           # Database Systems → Intro to Databases
    ("C016", "C004"),           # Frontend Dev → Web Fundamentals
    ("C017", "C006"),           # Operating Systems → Computer Systems
    ("C017", "C001"),           # Operating Systems → Intro to Programming
    ("C018", "C009"),           # Network Fundamentals → Intro to Networking
    ("C018", "C006"),           # Network Fundamentals → Computer Systems
    ("C019", "C001"),           # OOP → Intro to Programming
    ("C020", "C007"),           # Applied Statistics → Probability
    ("C020", "C010"),           # Applied Statistics → Linear Algebra
    ("C021", "C008"),           # UI/UX → UX Basics
    ("C021", "C004"),           # UI/UX → Web Fundamentals
    ("C022", "C001"),           # Systems Programming → Intro to Programming
    ("C022", "C006"),           # Systems Programming → Computer Systems

    # ── TIER 3 ────────────────────────────────────────────────────────────────
    ("C023", "C013"),           # Machine Learning → Statistical Inference
    ("C023", "C020"),           # Machine Learning → Applied Statistics
    ("C023", "C014"),           # Machine Learning → EDA
    ("C024", "C023"),           # Deep Learning → Machine Learning
    ("C024", "C012"),           # Deep Learning → Algorithms
    ("C025", "C019"),           # Software Engineering → OOP
    ("C025", "C012"),           # Software Engineering → Algorithms
    ("C026", "C017"),           # Distributed Systems → Operating Systems
    ("C026", "C018"),           # Distributed Systems → Network Fundamentals
    ("C027", "C015"),           # Data Pipelines → Database Systems
    ("C027", "C014"),           # Data Pipelines → EDA
    ("C027", "C022"),           # Data Pipelines → Systems Programming
    ("C028", "C024"),           # Computer Vision → Deep Learning
    ("C028", "C023"),           # Computer Vision → Machine Learning
    ("C029", "C023"),           # NLP → Machine Learning
    ("C029", "C013"),           # NLP → Statistical Inference
    ("C030", "C018"),           # Cloud Architecture → Network Fundamentals
    ("C030", "C026"),           # Cloud Architecture → Distributed Systems
    ("C031", "C018"),           # Cybersecurity → Network Fundamentals
    ("C031", "C017"),           # Cybersecurity → Operating Systems
    ("C032", "C019"),           # Mobile Dev → OOP
    ("C032", "C016"),           # Mobile Dev → Frontend Dev

    # ── TIER 4 ────────────────────────────────────────────────────────────────
    ("C033", "C024"),           # AI Capstone → Deep Learning
    ("C033", "C028"),           # AI Capstone → Computer Vision
    ("C033", "C029"),           # AI Capstone → NLP
    ("C034", "C025"),           # Full Stack Capstone → Software Engineering
    ("C034", "C016"),           # Full Stack Capstone → Frontend Dev
    ("C034", "C032"),           # Full Stack Capstone → Mobile Dev
    ("C035", "C027"),           # Data Engineering Capstone → Data Pipelines
    ("C035", "C030"),           # Data Engineering Capstone → Cloud Architecture
    ("C035", "C023"),           # Data Engineering Capstone → Machine Learning
    ("C036", "C026"),           # Systems Capstone → Distributed Systems
    ("C036", "C031"),           # Systems Capstone → Cybersecurity
    ("C037", "C030"),           # Cloud Capstone → Cloud Architecture
    ("C037", "C026"),           # Cloud Capstone → Distributed Systems
    ("C037", "C031"),           # Cloud Capstone → Cybersecurity
    ("C038", "C031"),           # Cybersecurity Capstone → Cybersecurity
    ("C038", "C026"),           # Cybersecurity Capstone → Distributed Systems
    ("C039", "C023"),           # Research Capstone → Machine Learning
    ("C039", "C024"),           # Research Capstone → Deep Learning
    ("C039", "C029"),           # Research Capstone → NLP
    ("C040", "C025"),           # Product Engineering → Software Engineering
    ("C040", "C032"),           # Product Engineering → Mobile Dev
    ("C040", "C021"),           # Product Engineering → UI/UX
]


# In[4]:


prereqs_df = pd.DataFrame(prereqs, columns=["course_id", "prereq_course_id"])
prereqs_df.to_csv("../datasets/course_prerequisites.csv", index=False)

print(f"✓ course_prerequisites.csv: {len(prereqs_df)} edges")
print(f"\nPrereqs per course:")
print(prereqs_df.course_id.value_counts().sort_values(ascending=False))


# In[ ]:




