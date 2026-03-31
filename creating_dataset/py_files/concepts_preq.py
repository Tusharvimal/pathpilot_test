#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


concept_prereqs = [
    # ── PROGRAMMING ───────────────────────────────────────────────────────────
    ("K003", "K001"),  # Functions → Variables
    ("K003", "K002"),  # Functions → Control Flow
    ("K004", "K003"),  # OOP → Functions
    ("K005", "K001"),  # Memory Management → Variables
    ("K009", "K004"),  # Design Patterns → OOP
    ("K010", "K003"),  # Testing → Functions
    ("K011", "K003"),  # Concurrency → Functions
    ("K011", "K005"),  # Concurrency → Memory Management
    ("K014", "K003"),  # Code Optimization → Functions
    ("K014", "K005"),  # Code Optimization → Memory Management
    ("K015", "K003"),  # Functional Programming → Functions
    ("K008", "K004"),  # APIs → OOP
    ("K012", "K008"),  # Data Serialization → APIs

    # ── MATHEMATICS ───────────────────────────────────────────────────────────
    ("K017", "K016"),  # Logic and Proofs → Set Theory
    ("K019", "K018"),  # Hypothesis Testing → Probability Distributions
    ("K019", "K017"),  # Hypothesis Testing → Logic and Proofs
    ("K021", "K020"),  # Matrix Operations → Linear Algebra Basics
    ("K023", "K022"),  # Gradient → Calculus Fundamentals
    ("K024", "K023"),  # Optimization → Gradient
    ("K024", "K021"),  # Optimization → Matrix Operations
    ("K026", "K018"),  # Bayesian → Probability Distributions
    ("K026", "K019"),  # Bayesian → Hypothesis Testing
    ("K027", "K018"),  # Statistical Modeling → Probability
    ("K027", "K028"),  # Statistical Modeling → Regression
    ("K028", "K020"),  # Regression → Linear Algebra Basics
    ("K028", "K018"),  # Regression → Probability Distributions
    ("K029", "K021"),  # Dimensionality Reduction → Matrix Operations
    ("K029", "K024"),  # Dimensionality Reduction → Optimization
    ("K030", "K018"),  # Information Theory → Probability

    # ── DATA SCIENCE ──────────────────────────────────────────────────────────
    ("K033", "K031"),  # Feature Engineering → Data Cleaning
    ("K033", "K032"),  # Feature Engineering → Exploratory Analysis
    ("K034", "K027"),  # Model Evaluation → Statistical Modeling
    ("K035", "K034"),  # Overfitting → Model Evaluation
    ("K036", "K034"),  # Cross Validation → Model Evaluation
    ("K037", "K024"),  # Neural Networks → Optimization
    ("K037", "K021"),  # Neural Networks → Matrix Operations
    ("K038", "K037"),  # Backpropagation → Neural Networks
    ("K038", "K023"),  # Backpropagation → Gradient
    ("K039", "K038"),  # CNN → Backpropagation
    ("K039", "K037"),  # CNN → Neural Networks
    ("K040", "K038"),  # RNN → Backpropagation
    ("K041", "K039"),  # Transfer Learning → CNN
    ("K041", "K040"),  # Transfer Learning → RNN
    ("K042", "K031"),  # Data Pipelines → Data Cleaning
    ("K042", "K069"),  # Data Pipelines → ETL
    ("K043", "K042"),  # Model Deployment → Data Pipelines
    ("K044", "K040"),  # NLP → RNN
    ("K044", "K030"),  # NLP → Information Theory
    ("K045", "K037"),  # Reinforcement Learning → Neural Networks
    ("K045", "K024"),  # Reinforcement Learning → Optimization

    # ── SYSTEMS ───────────────────────────────────────────────────────────────
    ("K047", "K046"),  # Memory Allocation → Process Management
    ("K048", "K047"),  # File Systems → Memory Allocation
    ("K050", "K049"),  # TCP IP → Networking Protocols
    ("K051", "K050"),  # DNS → TCP IP
    ("K052", "K046"),  # Distributed Computing → Process Management
    ("K052", "K050"),  # Distributed Computing → TCP IP
    ("K053", "K052"),  # Load Balancing → Distributed Computing
    ("K054", "K052"),  # Fault Tolerance → Distributed Computing
    ("K055", "K046"),  # Containerization → Process Management
    ("K055", "K048"),  # Containerization → File Systems
    ("K056", "K055"),  # Cloud Services → Containerization
    ("K056", "K053"),  # Cloud Services → Load Balancing
    ("K057", "K056"),  # Microservices → Cloud Services
    ("K057", "K054"),  # Microservices → Fault Tolerance
    ("K058", "K049"),  # Security Protocols → Networking Protocols
    ("K059", "K058"),  # Encryption → Security Protocols
    ("K060", "K059"),  # Authentication → Encryption

    # ── DATABASES ─────────────────────────────────────────────────────────────
    ("K062", "K061"),  # SQL → Relational Models
    ("K063", "K062"),  # Normalization → SQL
    ("K064", "K062"),  # Indexing → SQL
    ("K065", "K063"),  # Transactions → Normalization
    ("K066", "K061"),  # NoSQL → Relational Models
    ("K067", "K064"),  # Query Optimization → Indexing
    ("K068", "K063"),  # Data Warehousing → Normalization
    ("K068", "K066"),  # Data Warehousing → NoSQL
    ("K069", "K062"),  # ETL → SQL
    ("K069", "K068"),  # ETL → Data Warehousing
    ("K070", "K059"),  # Database Security → Encryption
    ("K070", "K065"),  # Database Security → Transactions

    # ── DESIGN ────────────────────────────────────────────────────────────────
    ("K072", "K071"),  # Wireframing → User Research
    ("K073", "K072"),  # Prototyping → Wireframing
    ("K074", "K073"),  # Accessibility → Prototyping
    ("K075", "K072"),  # Responsive Design → Wireframing
    ("K078", "K073"),  # Interaction Design → Prototyping
    ("K078", "K076"),  # Interaction Design → Color Theory
    ("K079", "K073"),  # Usability Testing → Prototyping
    ("K079", "K071"),  # Usability Testing → User Research
    ("K080", "K078"),  # Design Systems → Interaction Design
    ("K080", "K074"),  # Design Systems → Accessibility
]


# In[3]:


cp_df = pd.DataFrame(concept_prereqs, columns=["concept_id", "prereq_concept_id"])
cp_df.to_csv("../datasets/concept_prerequisites.csv", index=False)


# In[4]:


print(f"✓ concept_prerequisites.csv: {len(cp_df)} edges")
print(f"\nPrereqs per concept (avg): {cp_df.concept_id.value_counts().mean():.1f}")
print(f"\nMost foundational concepts (appear most as prereqs):")
print(cp_df.prereq_concept_id.value_counts().head(8))


# In[ ]:




