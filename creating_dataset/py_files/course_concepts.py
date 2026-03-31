#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


course_concepts = [
    # ── TIER 1 ────────────────────────────────────────────────────────────────
    ("C001", "K001"),  # Intro to Programming → Variables and Data Types
    ("C001", "K002"),  # Intro to Programming → Control Flow
    ("C001", "K003"),  # Intro to Programming → Functions and Recursion
    ("C001", "K006"),  # Intro to Programming → Debugging Techniques
    ("C001", "K013"),  # Intro to Programming → Command Line Tools

    ("C002", "K016"),  # Discrete Math → Set Theory
    ("C002", "K017"),  # Discrete Math → Logic and Proofs
    ("C002", "K025"),  # Discrete Math → Graph Theory

    ("C003", "K031"),  # Data Literacy → Data Cleaning
    ("C003", "K032"),  # Data Literacy → Exploratory Analysis
    ("C003", "K018"),  # Data Literacy → Probability Distributions

    ("C004", "K001"),  # Web Fundamentals → Variables and Data Types
    ("C004", "K075"),  # Web Fundamentals → Responsive Design
    ("C004", "K072"),  # Web Fundamentals → Wireframing

    ("C005", "K061"),  # Intro to Databases → Relational Models
    ("C005", "K062"),  # Intro to Databases → SQL Queries

    ("C006", "K046"),  # Computer Systems → Process Management
    ("C006", "K047"),  # Computer Systems → Memory Allocation
    ("C006", "K048"),  # Computer Systems → File Systems

    ("C007", "K018"),  # Probability → Probability Distributions
    ("C007", "K019"),  # Probability → Hypothesis Testing
    ("C007", "K026"),  # Probability → Bayesian Reasoning

    ("C008", "K071"),  # UX Basics → User Research
    ("C008", "K072"),  # UX Basics → Wireframing
    ("C008", "K076"),  # UX Basics → Color Theory
    ("C008", "K077"),  # UX Basics → Typography

    ("C009", "K049"),  # Intro to Networking → Networking Protocols
    ("C009", "K050"),  # Intro to Networking → TCP IP Stack

    ("C010", "K020"),  # Linear Algebra → Linear Algebra Basics
    ("C010", "K021"),  # Linear Algebra → Matrix Operations
    ("C010", "K022"),  # Linear Algebra → Calculus Fundamentals

    # ── TIER 2 ────────────────────────────────────────────────────────────────
    ("C011", "K003"),  # Data Structures → Functions and Recursion
    ("C011", "K005"),  # Data Structures → Memory Management
    ("C011", "K014"),  # Data Structures → Code Optimization
    ("C011", "K025"),  # Data Structures → Graph Theory

    ("C012", "K014"),  # Algorithms → Code Optimization
    ("C012", "K024"),  # Algorithms → Optimization Theory
    ("C012", "K025"),  # Algorithms → Graph Theory
    ("C012", "K015"),  # Algorithms → Functional Programming

    ("C013", "K019"),  # Statistical Inference → Hypothesis Testing
    ("C013", "K027"),  # Statistical Inference → Statistical Modeling
    ("C013", "K028"),  # Statistical Inference → Regression Analysis
    ("C013", "K026"),  # Statistical Inference → Bayesian Reasoning

    ("C014", "K031"),  # EDA → Data Cleaning
    ("C014", "K032"),  # EDA → Exploratory Analysis
    ("C014", "K033"),  # EDA → Feature Engineering
    ("C014", "K027"),  # EDA → Statistical Modeling

    ("C015", "K061"),  # Database Systems → Relational Models
    ("C015", "K063"),  # Database Systems → Database Normalization
    ("C015", "K064"),  # Database Systems → Indexing and Performance
    ("C015", "K065"),  # Database Systems → Transactions and ACID

    ("C016", "K075"),  # Frontend Dev → Responsive Design
    ("C016", "K078"),  # Frontend Dev → Interaction Design
    ("C016", "K008"),  # Frontend Dev → APIs and Interfaces
    ("C016", "K073"),  # Frontend Dev → Prototyping

    ("C017", "K046"),  # Operating Systems → Process Management
    ("C017", "K047"),  # Operating Systems → Memory Allocation
    ("C017", "K011"),  # Operating Systems → Concurrency and Threads
    ("C017", "K048"),  # Operating Systems → File Systems

    ("C018", "K049"),  # Network Fundamentals → Networking Protocols
    ("C018", "K050"),  # Network Fundamentals → TCP IP Stack
    ("C018", "K051"),  # Network Fundamentals → DNS and Routing
    ("C018", "K058"),  # Network Fundamentals → Security Protocols

    ("C019", "K004"),  # OOP → Object Oriented Design
    ("C019", "K009"),  # OOP → Design Patterns
    ("C019", "K010"),  # OOP → Testing and Validation
    ("C019", "K008"),  # OOP → APIs and Interfaces

    ("C020", "K027"),  # Applied Statistics → Statistical Modeling
    ("C020", "K028"),  # Applied Statistics → Regression Analysis
    ("C020", "K029"),  # Applied Statistics → Dimensionality Reduction
    ("C020", "K023"),  # Applied Statistics → Gradient and Derivatives

    ("C021", "K071"),  # UI/UX → User Research
    ("C021", "K073"),  # UI/UX → Prototyping
    ("C021", "K078"),  # UI/UX → Interaction Design
    ("C021", "K079"),  # UI/UX → Usability Testing
    ("C021", "K080"),  # UI/UX → Design Systems

    ("C022", "K005"),  # Systems Programming → Memory Management
    ("C022", "K011"),  # Systems Programming → Concurrency and Threads
    ("C022", "K046"),  # Systems Programming → Process Management
    ("C022", "K014"),  # Systems Programming → Code Optimization

    # ── TIER 3 ────────────────────────────────────────────────────────────────
    ("C023", "K033"),  # Machine Learning → Feature Engineering
    ("C023", "K034"),  # Machine Learning → Model Evaluation
    ("C023", "K035"),  # Machine Learning → Overfitting and Regularization
    ("C023", "K036"),  # Machine Learning → Cross Validation
    ("C023", "K028"),  # Machine Learning → Regression Analysis

    ("C024", "K037"),  # Deep Learning → Neural Networks
    ("C024", "K038"),  # Deep Learning → Backpropagation
    ("C024", "K035"),  # Deep Learning → Overfitting and Regularization
    ("C024", "K023"),  # Deep Learning → Gradient and Derivatives
    ("C024", "K041"),  # Deep Learning → Transfer Learning

    ("C025", "K009"),  # Software Engineering → Design Patterns
    ("C025", "K010"),  # Software Engineering → Testing and Validation
    ("C025", "K007"),  # Software Engineering → Version Control
    ("C025", "K043"),  # Software Engineering → Model Deployment

    ("C026", "K052"),  # Distributed Systems → Distributed Computing
    ("C026", "K053"),  # Distributed Systems → Load Balancing
    ("C026", "K054"),  # Distributed Systems → Fault Tolerance
    ("C026", "K057"),  # Distributed Systems → Microservices

    ("C027", "K042"),  # Data Pipelines → Data Pipelines
    ("C027", "K068"),  # Data Pipelines → Data Warehousing
    ("C027", "K069"),  # Data Pipelines → ETL Processes
    ("C027", "K064"),  # Data Pipelines → Indexing and Performance

    ("C028", "K039"),  # Computer Vision → Convolutional Networks
    ("C028", "K037"),  # Computer Vision → Neural Networks
    ("C028", "K041"),  # Computer Vision → Transfer Learning
    ("C028", "K034"),  # Computer Vision → Model Evaluation

    ("C029", "K044"),  # NLP → Natural Language Processing
    ("C029", "K040"),  # NLP → Recurrent Networks
    ("C029", "K030"),  # NLP → Information Theory
    ("C029", "K033"),  # NLP → Feature Engineering

    ("C030", "K055"),  # Cloud Architecture → Containerization
    ("C030", "K056"),  # Cloud Architecture → Cloud Services
    ("C030", "K052"),  # Cloud Architecture → Distributed Computing
    ("C030", "K053"),  # Cloud Architecture → Load Balancing

    ("C031", "K058"),  # Cybersecurity → Security Protocols
    ("C031", "K059"),  # Cybersecurity → Encryption and Hashing
    ("C031", "K060"),  # Cybersecurity → Authentication Systems
    ("C031", "K070"),  # Cybersecurity → Database Security

    ("C032", "K004"),  # Mobile Dev → Object Oriented Design
    ("C032", "K075"),  # Mobile Dev → Responsive Design
    ("C032", "K008"),  # Mobile Dev → APIs and Interfaces
    ("C032", "K073"),  # Mobile Dev → Prototyping

    # ── TIER 4 ────────────────────────────────────────────────────────────────
    ("C033", "K037"),  # AI Capstone → Neural Networks
    ("C033", "K041"),  # AI Capstone → Transfer Learning
    ("C033", "K045"),  # AI Capstone → Reinforcement Learning
    ("C033", "K043"),  # AI Capstone → Model Deployment
    ("C033", "K044"),  # AI Capstone → Natural Language Processing

    ("C034", "K008"),  # Full Stack → APIs and Interfaces
    ("C034", "K010"),  # Full Stack → Testing and Validation
    ("C034", "K057"),  # Full Stack → Microservices
    ("C034", "K043"),  # Full Stack → Model Deployment
    ("C034", "K080"),  # Full Stack → Design Systems

    ("C035", "K042"),  # Data Eng Capstone → Data Pipelines
    ("C035", "K068"),  # Data Eng Capstone → Data Warehousing
    ("C035", "K069"),  # Data Eng Capstone → ETL Processes
    ("C035", "K056"),  # Data Eng Capstone → Cloud Services
    ("C035", "K067"),  # Data Eng Capstone → Query Optimization

    ("C036", "K052"),  # Systems Capstone → Distributed Computing
    ("C036", "K054"),  # Systems Capstone → Fault Tolerance
    ("C036", "K059"),  # Systems Capstone → Encryption and Hashing
    ("C036", "K011"),  # Systems Capstone → Concurrency and Threads

    ("C037", "K055"),  # Cloud Capstone → Containerization
    ("C037", "K056"),  # Cloud Capstone → Cloud Services
    ("C037", "K057"),  # Cloud Capstone → Microservices
    ("C037", "K054"),  # Cloud Capstone → Fault Tolerance

    ("C038", "K058"),  # Cybersecurity Capstone → Security Protocols
    ("C038", "K059"),  # Cybersecurity Capstone → Encryption and Hashing
    ("C038", "K060"),  # Cybersecurity Capstone → Authentication Systems
    ("C038", "K052"),  # Cybersecurity Capstone → Distributed Computing

    ("C039", "K034"),  # Research Capstone → Model Evaluation
    ("C039", "K036"),  # Research Capstone → Cross Validation
    ("C039", "K045"),  # Research Capstone → Reinforcement Learning
    ("C039", "K030"),  # Research Capstone → Information Theory

    ("C040", "K073"),  # Product Eng → Prototyping
    ("C040", "K079"),  # Product Eng → Usability Testing
    ("C040", "K080"),  # Product Eng → Design Systems
    ("C040", "K043"),  # Product Eng → Model Deployment
]


# In[3]:


cc_df = pd.DataFrame(course_concepts, columns=["course_id", "concept_id"])
cc_df.to_csv("../datasets/course_concepts.csv", index=False)


# In[4]:


print(f"✓ course_concepts.csv: {len(cc_df)} edges")
print(f"\nConcepts per course (avg): {cc_df.course_id.value_counts().mean():.1f}")
print(f"\nMost covered concepts:")
print(cc_df.concept_id.value_counts().head(10))


# In[ ]:




