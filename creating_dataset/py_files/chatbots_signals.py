#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


np.random.seed(42)


# In[3]:


students     = pd.read_csv("../datasets/students.csv")
concepts     = pd.read_csv("../datasets/concepts.csv")
enrollments  = pd.read_csv("../datasets/enrollments.csv")
course_concepts = pd.read_csv("../datasets/course_concepts.csv")


# In[4]:


concept_difficulty_map = concepts.set_index("concept_id")["difficulty"].to_dict()
student_map = students.set_index("student_id").to_dict("index")
course_concept_map = course_concepts.groupby("course_id")["concept_id"].apply(list).to_dict()


# In[5]:


modality_help_style = {
    "visual":       "diagram",
    "analytical":   "explanation",
    "hands_on":     "practice",
    "exploratory":  "examples",
}


# In[6]:


chatbot_signals = []

for _, student in students.iterrows():
    student_id = student['student_id']
    ability = student['base_ability']
    effort = student['effort_level']
    risk = student['risk_factor']
    modality = student['true_modality']
    help_style = modality_help_style[modality]

    student_courses = enrollments[
        enrollments['student_id'] == student_id
    ]['course_id'].tolist()

    engaged_concepts = []
    for course_id in student_courses:
        concepts_in_course = course_concept_map.get(course_id, [])
        engaged_concepts.extend(concepts_in_course)

    engaged_concepts = list(set(engaged_concepts))

    engagement_rate = 0.20 + effort * 0.20
    engaged_concepts = [
        c for c in engaged_concepts
        if np.random.random() < engagement_rate
    ]

    for concept_id in engaged_concepts:
        concept_diff = concept_difficulty_map[concept_id]

        confusion_raw = concept_diff * (1 - ability) * 1.5 + np.random.normal(0, 0.08)
        confusion_score = round(np.clip(confusion_raw, 0, 1), 3)

        avg_questions = 2 + confusion_score * 8 + (1 - ability) * 2
        num_questions = max(1, int(np.random.poisson(avg_questions)))

        help_rate = 0.2 + confusion_score * 0.4
        help_requests = max(0, int(num_questions * help_rate))

        avg_recency = 30 - effort * 20
        recency = max(1, int(np.random.normal(avg_recency, 5)))

        chatbot_signals.append({
            "student_id":      student_id,
            "concept_id":      concept_id,
            "num_questions":   num_questions,
            "confusion_score": confusion_score,
            "help_requests":   help_requests,
            "recency_days":    recency,
            "help_style":      help_style,
        })


# In[7]:


chatbot_df = pd.DataFrame(chatbot_signals)
chatbot_df.to_csv("../datasets/chatbot_signals.csv", index=False)


# In[8]:


print(f"✓ chatbot_signals.csv: {len(chatbot_df)} rows")
print(f"\nAvg concepts per student: {len(chatbot_df) / len(students):.1f}")
print(f"\nConfusion score distribution:")
print(pd.cut(chatbot_df.confusion_score, bins=[0,.25,.5,.75,1],
    labels=["low","med","high","very_high"]).value_counts().sort_index())
print(f"\nHelp style distribution:\n{chatbot_df.help_style.value_counts()}")
print(f"\nAvg questions per concept: {chatbot_df.num_questions.mean():.1f}")


# In[ ]:




