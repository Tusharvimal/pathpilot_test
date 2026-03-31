#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


enrollments     = pd.read_csv("../datasets/enrollments.csv")
chatbot         = pd.read_csv("../datasets/chatbot_signals.csv")
course_prereqs  = pd.read_csv("../datasets/course_prerequisites.csv")
course_concepts = pd.read_csv("../datasets/course_concepts.csv")
concept_prereqs = pd.read_csv("../datasets/concept_prerequisites.csv")


# In[3]:


all_edges = []

for _, row in enrollments.iterrows():
    all_edges.append({
        "src_id":    row["student_id"],
        "dst_id":    row["course_id"],
        "edge_type": "enrolled_in",
        "weight":    round(row["grade"] / 4.0, 4),
    })

for _, row in chatbot.iterrows():
    all_edges.append({
        "src_id":    row["student_id"],
        "dst_id":    row["concept_id"],
        "edge_type": "engaged_with",
        "weight":    round(row["confusion_score"], 4),
    })

for _, row in course_prereqs.iterrows():
    all_edges.append({
        "src_id":    row["course_id"],
        "dst_id":    row["prereq_course_id"],
        "edge_type": "prereq_of",
        "weight":    1.0,
    })

for _, row in course_concepts.iterrows():
    all_edges.append({
        "src_id":    row["course_id"],
        "dst_id":    row["concept_id"],
        "edge_type": "covers",
        "weight":    1.0,
    })

for _, row in concept_prereqs.iterrows():
    all_edges.append({
        "src_id":    row["concept_id"],
        "dst_id":    row["prereq_concept_id"],
        "edge_type": "concept_prereq",
        "weight":    1.0,
    })


# In[4]:


edges_df = pd.DataFrame(all_edges)
edges_df.to_csv("../datasets/graph_edges.csv", index=False)


# In[5]:


print(f"✓ graph_edges.csv: {len(edges_df)} rows")
print(f"\nEdges per type:\n{edges_df.edge_type.value_counts()}")
print(f"\nWeight distribution for enrolled_in edges:")
enrolled = edges_df[edges_df.edge_type == "enrolled_in"]
print(pd.cut(enrolled.weight, bins=[0,.25,.5,.75,1],
    labels=["weak","below_avg","above_avg","strong"]).value_counts().sort_index())


# In[ ]:




