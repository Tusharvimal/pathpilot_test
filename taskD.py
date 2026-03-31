#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import GraphSAGE


# In[2]:


checkpoint = torch.load('models/task_a_model.pt', weights_only=False)
data = checkpoint['data']
student_id_map = checkpoint['student_id_map']
course_id_map = checkpoint['course_id_map']
concept_id_map = checkpoint['concept_id_map']

model = GraphSAGE(hidden_dim=256)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

enrollments = pd.read_csv('datasets/enrollments.csv')
courses = pd.read_csv('datasets/courses.csv')
chatbot = pd.read_csv('datasets/chatbot_signals.csv')
assessments = pd.read_csv('datasets/assessment_scores.csv')
course_concepts_df = pd.read_csv('datasets/course_concepts.csv')
concepts = pd.read_csv('datasets/concepts.csv')


# In[3]:


course_name_map = courses.set_index('course_id')['course_name'].to_dict()
concept_name_map = concepts.set_index('concept_id')['concept_name'].to_dict()

failing = enrollments[enrollments['grade'] < 1.5]

course_stats = enrollments.groupby('course_id').agg(
    total_enrolled = ('student_id', 'count'),
    avg_grade = ('grade', 'mean'),
    fail_count = ('passed', lambda x: (x==0).sum()),
    fail_rate = ('passed', lambda x: (x==0).mean()),
).reset_index()

course_stats['course_name'] = course_stats['course_id'].map(course_name_map)
course_stats = course_stats.sort_values('fail_rate', ascending = False)

print("Top 10 courses by failure rate: ")
print(course_stats[['course_name', 'total_enrolled', 'fail_count', 'fail_rate', 'avg_grade']].head(10).to_string(index = False))


# In[4]:


failing_students = failing['student_id'].unique()

failing_chatbot = chatbot[chatbot['student_id'].isin(failing_students)]

concept_to_course = course_concepts_df.groupby('concept_id')['course_id'].apply(list).to_dict()

results = []
for course_id in course_stats['course_id']:
    course_concepts = course_concepts_df[course_concepts_df['course_id'] == course_id]['concept_id'].tolist()

    course_failing = failing[failing['course_id'] == course_id]['student_id'].tolist()

    relevant = chatbot[
        (chatbot['student_id'].isin(course_failing)) &
        (chatbot['concept_id'].isin(course_concepts))
    ]

    if len(relevant) > 0:
        concept_confusion = relevant.groupby('concept_id')['confusion_score'].mean().sort_values(ascending=False)

        for concept_id, confusion in concept_confusion.head(3).items():
            results.append({
                'course': course_name_map[course_id],
                'concept': concept_name_map[concept_id], 
                'avg_confusion': round(confusion, 3)
            })

results_df = pd.DataFrame(results)
print("Top 3 most confusing concepts per struggling course:")
print(results_df.to_string(index=False))


# In[6]:


model.eval()
student_indices = torch.tensor([student_id_map[s] for s in enrollments['student_id']], dtype = torch.long)
course_indices = torch.tensor([course_id_map[c] for c in enrollments['course_id']], dtype = torch.long)

with torch.no_grad():
    out = model(data)
    student_embs = out[0][student_indices]
    course_embs = out[1][course_indices]
    pairs = torch.cat([student_embs, course_embs, student_embs - course_embs, student_embs * course_embs], dim = 1)
    baseline_preds = model.task_a_head(pairs).argmax(dim = 1)

print(f"Baseline total High risk: {(baseline_preds == 2).sum().item()}")


# In[8]:


print("=== Counterfactual: Reduce avg_confusion by 50% for High-risk students ===\n")

model.eval()

# Get baseline predictions
with torch.no_grad():
    out = model(data)
    pairs = torch.cat([out[0][student_indices], out[1][course_indices], 
                       out[0][student_indices] - out[1][course_indices],
                       out[0][student_indices] * out[1][course_indices]], dim=1)
    baseline_preds = model.task_a_head(pairs).argmax(dim=1)

# Save original student features
original_student_x = data['student'].x.clone()

# avg_confusion is the 4th feature (index 3) after gpa_start, effort_level, risk_factor
# Reduce it by 50% for ALL students
data['student'].x[:, 3] = data['student'].x[:, 3] * 0.5

# Run model again
with torch.no_grad():
    out = model(data)
    pairs = torch.cat([out[0][student_indices], out[1][course_indices],
                       out[0][student_indices] - out[1][course_indices],
                       out[0][student_indices] * out[1][course_indices]], dim=1)
    new_preds = model.task_a_head(pairs).argmax(dim=1)

# Compare per course
for course_id in course_stats['course_id'].head(10):
    course_mask = (enrollments['course_id'] == course_id).values
    baseline_count = (baseline_preds[course_mask] == 2).sum().item()
    new_count = (new_preds[course_mask] == 2).sum().item()

    print(f"{course_name_map[course_id]}:")
    print(f"  Baseline High-risk: {baseline_count}")
    print(f"  After reducing confusion: {new_count}")
    print(f"  Students helped: {baseline_count - new_count}")
    print()

# Restore original
data['student'].x = original_student_x


# In[9]:


print("=== Course Improvement Report ===\n")

total_baseline = 0
total_new = 0

for course_id in course_stats['course_id']:
    course_name = course_name_map[course_id]
    course_results = results_df[results_df['course'] == course_name]

    course_mask = (enrollments['course_id'] == course_id).values
    baseline_count = (baseline_preds[course_mask] == 2).sum().item()
    new_count = (new_preds[course_mask] == 2).sum().item()

    total_baseline += baseline_count
    total_new += new_count

print(f"Total High-risk students (baseline): {total_baseline}")
print(f"Total High-risk students (after intervention): {total_new}")
print(f"Total students helped: {total_baseline - total_new}")
print(f"Reduction: {(total_baseline - total_new) / total_baseline:.1%}")


# In[10]:


print("=== Counterfactual: Targeted concept tutoring per course ===\n")

chatbot = pd.read_csv('datasets/chatbot_signals.csv')

for course_id in course_stats['course_id'].head(10):
    course_name = course_name_map[course_id]
    course_results = results_df[results_df['course'] == course_name]

    if len(course_results) == 0:
        continue

    reverse_concept_name = {v: k for k, v in concept_name_map.items()}
    top3_concepts = [reverse_concept_name[name] for name in course_results['concept'].values]

    # Find students who interacted with these concepts
    confused_students = chatbot[chatbot['concept_id'].isin(top3_concepts)]['student_id'].unique()
    confused_student_indices = [student_id_map[s] for s in confused_students if s in student_id_map]

    # Save original
    original_student_x = data['student'].x.clone()

    # Reduce avg_confusion ONLY for these students
    for s_idx in confused_student_indices:
        data['student'].x[s_idx, 3] = data['student'].x[s_idx, 3] * 0.5

    # Run model
    with torch.no_grad():
        out = model(data)
        pairs = torch.cat([out[0][student_indices], out[1][course_indices],
                           out[0][student_indices] - out[1][course_indices],
                           out[0][student_indices] * out[1][course_indices]], dim=1)
        new_preds = model.task_a_head(pairs).argmax(dim=1)

    course_mask = (enrollments['course_id'] == course_id).values
    baseline_count = (baseline_preds[course_mask] == 2).sum().item()
    new_count = (new_preds[course_mask] == 2).sum().item()

    concept_names = [concept_name_map[c] for c in top3_concepts]

    print(f"{course_name}:")
    print(f"  Tutoring target: {', '.join(concept_names)}")
    print(f"  Students receiving tutoring: {len(confused_student_indices)}")
    print(f"  Baseline High-risk: {baseline_count}")
    print(f"  After targeted tutoring: {new_count}")
    print(f"  Students helped: {baseline_count - new_count}")
    print()

    # Restore
    data['student'].x = original_student_x


# In[ ]:




