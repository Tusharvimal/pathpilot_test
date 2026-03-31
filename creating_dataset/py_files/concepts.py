#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
np.random.seed(42)

concepts_raw = [
    # Programming — 15 concepts
    ("K001", "Variables and Data Types",        "programming"),
    ("K002", "Control Flow",                    "programming"),
    ("K003", "Functions and Recursion",         "programming"),
    ("K004", "Object Oriented Design",          "programming"),
    ("K005", "Memory Management",               "programming"),
    ("K006", "Debugging Techniques",            "programming"),
    ("K007", "Version Control",                 "programming"),
    ("K008", "APIs and Interfaces",             "programming"),
    ("K009", "Design Patterns",                 "programming"),
    ("K010", "Testing and Validation",          "programming"),
    ("K011", "Concurrency and Threads",         "programming"),
    ("K012", "Data Serialization",              "programming"),
    ("K013", "Command Line Tools",              "programming"),
    ("K014", "Code Optimization",               "programming"),
    ("K015", "Functional Programming",          "programming"),

    # Mathematics — 15 concepts
    ("K016", "Set Theory",                      "mathematics"),
    ("K017", "Logic and Proofs",                "mathematics"),
    ("K018", "Probability Distributions",       "mathematics"),
    ("K019", "Hypothesis Testing",              "mathematics"),
    ("K020", "Linear Algebra Basics",           "mathematics"),
    ("K021", "Matrix Operations",               "mathematics"),
    ("K022", "Calculus Fundamentals",           "mathematics"),
    ("K023", "Gradient and Derivatives",        "mathematics"),
    ("K024", "Optimization Theory",             "mathematics"),
    ("K025", "Graph Theory",                    "mathematics"),
    ("K026", "Bayesian Reasoning",              "mathematics"),
    ("K027", "Statistical Modeling",            "mathematics"),
    ("K028", "Regression Analysis",             "mathematics"),
    ("K029", "Dimensionality Reduction",        "mathematics"),
    ("K030", "Information Theory",              "mathematics"),

    # Data Science — 15 concepts
    ("K031", "Data Cleaning",                   "data_science"),
    ("K032", "Exploratory Analysis",            "data_science"),
    ("K033", "Feature Engineering",             "data_science"),
    ("K034", "Model Evaluation",                "data_science"),
    ("K035", "Overfitting and Regularization",  "data_science"),
    ("K036", "Cross Validation",                "data_science"),
    ("K037", "Neural Networks",                 "data_science"),
    ("K038", "Backpropagation",                 "data_science"),
    ("K039", "Convolutional Networks",          "data_science"),
    ("K040", "Recurrent Networks",              "data_science"),
    ("K041", "Transfer Learning",               "data_science"),
    ("K042", "Data Pipelines",                  "data_science"),
    ("K043", "Model Deployment",                "data_science"),
    ("K044", "Natural Language Processing",     "data_science"),
    ("K045", "Reinforcement Learning",          "data_science"),

    # Systems — 15 concepts
    ("K046", "Process Management",              "systems"),
    ("K047", "Memory Allocation",               "systems"),
    ("K048", "File Systems",                    "systems"),
    ("K049", "Networking Protocols",            "systems"),
    ("K050", "TCP IP Stack",                    "systems"),
    ("K051", "DNS and Routing",                 "systems"),
    ("K052", "Distributed Computing",           "systems"),
    ("K053", "Load Balancing",                  "systems"),
    ("K054", "Fault Tolerance",                 "systems"),
    ("K055", "Containerization",                "systems"),
    ("K056", "Cloud Services",                  "systems"),
    ("K057", "Microservices",                   "systems"),
    ("K058", "Security Protocols",              "systems"),
    ("K059", "Encryption and Hashing",          "systems"),
    ("K060", "Authentication Systems",          "systems"),

    # Databases — 10 concepts
    ("K061", "Relational Models",               "databases"),
    ("K062", "SQL Queries",                     "databases"),
    ("K063", "Database Normalization",          "databases"),
    ("K064", "Indexing and Performance",        "databases"),
    ("K065", "Transactions and ACID",           "databases"),
    ("K066", "NoSQL Databases",                 "databases"),
    ("K067", "Query Optimization",              "databases"),
    ("K068", "Data Warehousing",                "databases"),
    ("K069", "ETL Processes",                   "databases"),
    ("K070", "Database Security",               "databases"),

    # Design — 10 concepts
    ("K071", "User Research",                   "design"),
    ("K072", "Wireframing",                     "design"),
    ("K073", "Prototyping",                     "design"),
    ("K074", "Accessibility Standards",         "design"),
    ("K075", "Responsive Design",               "design"),
    ("K076", "Color Theory",                    "design"),
    ("K077", "Typography",                      "design"),
    ("K078", "Interaction Design",              "design"),
    ("K079", "Usability Testing",               "design"),
    ("K080", "Design Systems",                  "design"),
]

concepts = pd.DataFrame(concepts_raw,
    columns=["concept_id", "concept_name", "domain"])

domain_difficulty = {
    "programming": 0.45,
    "mathematics":  0.65,
    "data_science": 0.60,
    "systems":      0.55,
    "databases":    0.50,
    "design":       0.35,
}

difficulty = []
for domain in concepts["domain"]:
    base  = domain_difficulty[domain]
    noise = np.random.uniform(-0.10, 0.10)
    difficulty.append(round(np.clip(base + noise, 0.1, 0.95), 2))

concepts["difficulty"] = difficulty

concepts.to_csv("../datasets/concepts.csv", index=False)

print(f"✓ concepts.csv: {len(concepts)} rows")
print(f"\nConcepts per domain:\n{concepts.domain.value_counts()}")
print(f"\nDifficulty by domain:")
print(concepts.groupby("domain")["difficulty"].agg(["min", "mean", "max"]).round(2))


# In[ ]:




