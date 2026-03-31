#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler


# In[2]:


DATA_DIR = "datasets"
OUTPUT_PATH = "datasets/graph.pt" 


# In[3]:


students = pd.read_csv(f"{DATA_DIR}/students.csv")
courses = pd.read_csv(f"{DATA_DIR}/courses.csv")
course_prerequisites = pd.read_csv(f"{DATA_DIR}/course_prerequisites.csv")
concepts = pd.read_csv(f"{DATA_DIR}/concepts.csv")
concept_prerequisites = pd.read_csv(f"{DATA_DIR}/concept_prerequisites.csv")
course_concepts = pd.read_csv(f"{DATA_DIR}/course_concepts.csv")
enrollments = pd.read_csv(f"{DATA_DIR}/enrollments.csv")
assessment_scores = pd.read_csv(f"{DATA_DIR}/assessment_scores.csv")
chatbot_signals = pd.read_csv(f"{DATA_DIR}/chatbot_signals.csv")
graph_edges = pd.read_csv(f"{DATA_DIR}/graph_edges.csv")


# In[4]:


for name, df in [("students", students), ("courses", courses), ("concepts", concepts), 
                  ("enrollments", enrollments), ("chatbot_signals", chatbot_signals)]:
    print(f"{name}: {df.shape}  |  cols: {list(df.columns)}")


# In[5]:


def normalize(df, cols):
    scaler = StandardScaler()
    array = scaler.fit_transform(df[cols].values)
    array_x = torch.tensor(array, dtype = torch.float)

    print(array_x.shape)
    return array_x


# In[ ]:


students_cols = ['gpa_start', 'effort_level']
courses_cols = ['tier', 'difficulty', 'pass_rate']
concepts_cols = ['difficulty']

students_x = normalize(students, students_cols)
courses_x = normalize(courses, courses_cols)
concepts_x = normalize(concepts, concepts_cols)


# In[7]:


print(students_x)


# In[8]:


student_id_map = {sid: idx for idx, sid in enumerate(students['student_id'])}
course_id_map = {cid: idx for idx, cid in enumerate(courses['course_id'])}
concept_id_map = {cid: idx for idx, cid in enumerate(concepts['concept_id'])}


# In[9]:


print(graph_edges['edge_type'].unique())


# In[10]:


# print(len(enrolled_edges))
# print(len(engaged_edges))
# print(len(prereq_edges))
# print(len(covers_edges))
# print(len(concept_prereq_edges))


# In[11]:


def making_index(src_map, dest_map, edges):
    src = [src_map[i] for i in edges['src_id']]
    dst = [dest_map[i] for i in edges['dst_id']]

    edge_index = torch.tensor([src, dst], dtype = torch.long)
    edge_weight = torch.tensor(edges['weight'].values, dtype=torch.float)

    print(edge_index.shape, edge_weight.shape)

    return edge_index, edge_weight


# In[12]:


enrolled_edges = graph_edges[graph_edges['edge_type'] == 'enrolled_in']
engaged_edges = graph_edges[graph_edges['edge_type'] == 'engaged_with']
prereq_edges = graph_edges[graph_edges['edge_type'] == 'prereq_of']
covers_edges = graph_edges[graph_edges['edge_type'] == 'covers']
concept_prereq_edges = graph_edges[graph_edges['edge_type'] == 'concept_prereq']


# In[13]:


enrolled_edge_index, enrolled_edge_weight = making_index(student_id_map, course_id_map, enrolled_edges)
engaged_edge_index, engaged_edge_weight = making_index(student_id_map, concept_id_map, engaged_edges)
prereq_edge_index, prereq_edge_weight = making_index(course_id_map, course_id_map, prereq_edges)
cover_edge_index, cover_edge_weight = making_index(course_id_map, concept_id_map, covers_edges)
concept_prereq_edge_index, concept_prereq_edge_weight = making_index(concept_id_map, concept_id_map, concept_prereq_edges)


# In[27]:


data = HeteroData()

data['student'].x = students_x
data['course'].x = courses_x
data['concept'].x = concepts_x

data['student', 'enrolled_in', 'course'].edge_index = enrolled_edge_index
data['student', 'enrolled_in', 'course'].edge_weight = enrolled_edge_weight

data['student', 'engaged_with', 'concept'].edge_index = engaged_edge_index
data['student', 'engaged_with', 'concept'].edge_weight = engaged_edge_weight

data['course', 'prereq_of', 'course'].edge_index = prereq_edge_index
data['course', 'prereq_of', 'course'].edge_weight = prereq_edge_weight

data['course', 'covers', 'concept'].edge_index = cover_edge_index
data['course', 'covers', 'concept'].edge_weight = cover_edge_weight

data['concept', 'concept_prereq', 'concept'].edge_index = concept_prereq_edge_index
data['concept', 'concept_prereq', 'concept'].edge_weight = concept_prereq_edge_weight

data['course', 'rev_enrolled_in', 'student'].edge_index = enrolled_edge_index.flip(0)
data['concept','rev_engaged_with', 'student'].edge_index = engaged_edge_index.flip(0)



# In[28]:


print(data)


# In[29]:


train_mask = torch.tensor(students['split'] == 'train', dtype = torch.bool)
val_mask = torch.tensor(students['split'] == 'val', dtype = torch.bool)
test_mask = torch.tensor(students['split'] == 'test', dtype = torch.bool)

data['student'].train_mask = train_mask
data['student'].val_mask = val_mask
data['student'].test_mask = test_mask

print(train_mask.sum())
print(val_mask.sum())
print(test_mask.sum())


# In[ ]:


## TSNE

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv1 = nn.ModuleDict({
            'enrolled_in': SAGEConv(-1, hidden_dim),
            'engaged_with': SAGEConv(-1, hidden_dim),
            'prereq_of': SAGEConv(-1, hidden_dim),
            'covers': SAGEConv(-1, hidden_dim),
            'concept_prereq': SAGEConv(-1, hidden_dim)
        })

        self.conv2 = nn.ModuleDict({
            'rev_enrolled_in': SAGEConv(hidden_dim, hidden_dim),
            'rev_engaged_with': SAGEConv(hidden_dim, hidden_dim)
        })

        self.task_a_head = nn.Linear(hidden_dim*2, 1)

    def forward(self, data):

        x_student = data['student'].x
        x_course = data['course'].x
        x_concept = data['concept'].x

        ## Layer 1 
        course_from_students = self.conv1['enrolled_in'](
            (x_student, x_course),
            data['student', 'enrolled_in', 'course'].edge_index
        )
        course_from_course = self.conv1['prereq_of'](
            (x_course, x_course),
            data['course', 'prereq_of', 'course'].edge_index
        )
        concepts_from_students = self.conv1['engaged_with'](
            (x_student, x_concept),
            data['student', 'engaged_with', 'concept'].edge_index
        )
        concepts_from_course = self.conv1['covers'](
            (x_course, x_concept),
            data['course', 'covers', 'concept'].edge_index
        )
        concepts_from_concepts = self.conv1['concept_prereq'](
            (x_concept, x_concept),
            data['concept', 'concept_prereq', 'concept'].edge_index
        )

        # Relu
        x_course_1 = torch.relu(course_from_students + course_from_course)
        x_concept_1 = torch.relu(concepts_from_students + concepts_from_course + concepts_from_concepts)

        ## Layer 2
        student_from_course = self.conv2['rev_enrolled_in'](
            (x_course_1, None),
            data['course', 'rev_enrolled_in', 'student'].edge_index,
            size=(x_course_1.size(0), x_student.size(0))   # ← (num_src, num_dst)
        )

        student_from_concept = self.conv2['rev_engaged_with'](
            (x_concept_1, None),
            data['concept', 'rev_engaged_with', 'student'].edge_index,
            size=(x_concept_1.size(0), x_student.size(0))   # ← (num_src, num_dst)
        )

        x_student_1 = torch.relu(student_from_course + student_from_concept)

        return x_student_1, x_course_1, x_concept_1

model = GraphSAGE(hidden_dim=64)
out = model(data)
print(out[0].shape)  # should be (2000, 64)
print(out[1].shape)  # should be (40, 64)
print(out[2].shape)  # should be (80, 64)


# In[37]:


student_indices = torch.tensor(
    [student_id_map[s] for s in enrollments['student_id']],
    dtype = torch.long
)

course_indices = torch.tensor(
    [course_id_map[c] for c in enrollments['course_id']],
    dtype= torch.long
)

labels = torch.tensor(
    enrollments['grade'].values / 4.0,
    dtype = torch.float
)

train_mask_enrollments = train_mask[student_indices]

print(student_indices.shape, course_indices.shape, labels.shape, train_mask_enrollments.shape, train_mask_enrollments.sum())


# In[38]:


student_embs = out[0][student_indices]

print(student_embs.shape)


# In[47]:


optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = torch.nn.L1Loss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    out = model(data)

    student_embs = out[0][student_indices]
    course_embs = out[1][course_indices]

    pairs = torch.cat([student_embs, course_embs], dim = 1)

    preds = model.task_a_head(pairs).squeeze()
    loss = loss_fn(preds[train_mask_enrollments], labels[train_mask_enrollments])

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


# In[48]:


val_mask_enrollments = val_mask[student_indices]
test_mask_enrollments = test_mask[student_indices]


# In[49]:


def evaluate(model):
    model.eval()
    with torch.no_grad():
        out = model(data)

        student_embs = out[0][student_indices]
        course_embs = out[1][course_indices]

        pairs = torch.cat([student_embs, course_embs], dim = 1)

        preds = model.task_a_head(pairs).squeeze()
        loss = loss_fn(preds[val_mask_enrollments], labels[val_mask_enrollments])
        print(f" Val Loss: {loss.item():.4f}")

    val_preds  = preds[val_mask_enrollments]
    val_labels = labels[val_mask_enrollments]

    # print first 10 side by side
    for i in range(10):
        print(f"Predicted: {val_preds[i].item():.3f} | Actual: {val_labels[i].item():.3f}")

evaluate(model)


# In[50]:


def bin_risk(pred):
    if pred < 0.50:
        return 'Critical'
    elif pred < 0.75:
        return 'High'
    elif pred < 0.875:
        return 'Med'
    else:
        return 'Low'

# apply to val predictions
with torch.no_grad():
    out = model(data)
    student_embs = out[0][student_indices]
    course_embs  = out[1][course_indices]
    pairs = torch.cat([student_embs, course_embs], dim=1)
    preds = model.task_a_head(pairs).squeeze()

val_preds  = preds[val_mask_enrollments]
val_labels = labels[val_mask_enrollments]

for i in range(10):
    pred_bin = bin_risk(val_preds[i].item())
    actual_bin = bin_risk(val_labels[i].item())
    print(f"Predicted: {val_preds[i].item():.3f} ({pred_bin}) | Actual: {val_labels[i].item():.3f} ({actual_bin})")


# In[51]:


print("Predicted bins:")
pred_bins = [bin_risk(p.item()) for p in val_preds]
import collections
print(collections.Counter(pred_bins))

print("\nActual bins:")
actual_bins = [bin_risk(l.item()) for l in val_labels]
print(collections.Counter(actual_bins))


# In[ ]:




