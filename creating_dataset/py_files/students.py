#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd


# In[11]:


np.random.seed(42)
N = 2000


# ### Students Data

# In[12]:


base_ability = np.random.beta(5, 3, N)

effort_noise = np.random.beta(4,4, N)
effort_level = np.clip(0.4 * base_ability + 0.6*effort_noise, 0, 1)

modalities = ['visual', 'hands_on', 'exploratory', 'analytical']
true_modality = np.random.choice(modalities, N, p = [0.30, 0.25, 0.20, 0.25])

gpa_start = np.clip(
    base_ability * 2.2 + effort_level *1.5 + np.random.normal(0, 0.2, N), 0.0, 4.0
)

risk_raw = (1 - base_ability)* 0.45 + (1-effort_level)* 0.45 + 0.02
risk_factor = np.clip(risk_raw + np.random.normal(0, 0.08, N), 0, 1)

indices = np.arange(N)
np.random.shuffle(indices)

split = np.empty(N, dtype = 'object')
split[indices[:1400]] = "train"
split[indices[1400: 1700]] = "val"
split[indices[1700:]] = "test"


# In[13]:


students = pd.DataFrame({
    "student_id": [f"S{i:04d}" for i in range(N)],
    "base_ability": base_ability.round(4),
    "effort_level": effort_level.round(4),
    "true_modality": true_modality,
    "gpa_start": gpa_start.round(2),
    "risk_factor": risk_factor.round(4),
    "split": split
})

students.to_csv("../datasets/students.csv", index=False)


# In[14]:


print(f"✓ {len(students)} students saved")
print(f"\nSplit counts:\n{students.split.value_counts()}")


# ### Courses Data

# In[ ]:




