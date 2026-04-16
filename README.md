# PathPilot 🎓

A GNN-based student learning guidance system that uses a heterogeneous graph of students, courses, and concepts to predict student risk, recommend learning paths, match learning modalities, and suggest course improvements.

---

## Project Structure

```
PathPilot/
├── datasets/                        # CSV data files
│   ├── students.csv
│   ├── courses.csv
│   ├── enrollments.csv
│   ├── concepts.csv
│   ├── course_concepts.csv
│   ├── course_prerequisites.csv
│   ├── concept_prerequisites.csv
│   ├── concept_embeddings.npy
│   ├── course_embeddings.npy
│   ├── chatbot_signals.csv
│   ├── assessment_scores.csv
│   └── graph_edges.csv
├── models/                          # Saved model checkpoints
│   ├── task_a_model.pt
│   ├── task_a_model_new.pt
│   ├── task_b_link_model.pt
│   ├── task_b_link_model_new.pt
│   ├── task_c_model.pt
│   └── task_c_model_new.pt
├── creating_dataset/                # Notebooks to regenerate datasets
│   ├── students.ipynb
│   ├── courses.ipynb
│   ├── enrollments.ipynb
│   ├── assessments.ipynb
│   ├── chatbots_signals.ipynb
│   ├── concepts.ipynb
│   ├── concepts_preq.ipynb
│   ├── course_concepts.ipynb
│   ├── course_prerequisites.ipynb
│   ├── graph_edges.ipynb
│   └── embeddings.ipynb
├── frontend/                        # React frontend
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   ├── index.css
│   │   ├── StudentDashboard.js
│   │   ├── ProfessorDashboard.js
│   │   └── reportWebVitals.js
│   ├── public/
│   ├── package.json
│   └── .gitignore
├── model.py                         # GraphSAGE model definition
├── utils.py                         # Shared utilities
├── api.py                           # FastAPI backend server
├── build_graph.ipynb                # Graph construction + Task A training
├── taskB.ipynb                      # Task B — Concept Sequencing
├── taskC.ipynb                      # Task C — Learning Modality Match
├── taskD.ipynb                      # Task D — Course Improvement Recommender
├── phase3.ipynb                     # Cold start / real-time inferencing
├── dataset_analysis.ipynb           # EDA and t-SNE visualizations
├── requirements.txt
└── .gitignore
```

---

## What Each Task Does

### Task A — Student Success Navigator
Predicts a student's risk of failing a specific course **before** they take it. Risk is bucketed into Low / Medium / High. The same student will have different risk scores for different courses.

### Task B — Concept Sequencing Planner
For students flagged as high risk, identifies which concepts required by the course they are weak on and recommends an optimal learning sequence to fill those gaps using prerequisite chains.

### Task C — Learning Modality Match
Predicts a student's true learning modality (visual, hands-on, analytical, exploratory) and flags mismatches between the student's style and the course's delivery format.

### Task D — Course Improvement Recommender
Aggregates patterns across all struggling students in a course to identify which concepts are consistently confusing and what structural changes would help future cohorts.

---

## Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- Jupyter Notebook or JupyterLab

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd PathPilot
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Project

### Start the Backend (FastAPI)

From the root of the project:

```bash
uvicorn api:app --reload
```

The backend will be running at **http://localhost:8000**

You can view the auto-generated API docs at **http://localhost:8000/docs**

### Start the Frontend (React)

In a **separate terminal**:

```bash
cd frontend
npm start
```

The frontend will open at **http://localhost:3000**

---

## Running the Notebooks

Open Jupyter and run the notebooks in this order:

### 1. Train the main model (Task A)
```bash
jupyter notebook build_graph.ipynb
```
This trains the GraphSAGE model and saves the checkpoint to `models/task_a_model.pt`. **Run this before any other task notebook.**

### 2. Run individual tasks
```bash
jupyter notebook taskB.ipynb   # Concept Sequencing
jupyter notebook taskC.ipynb   # Learning Modality Match
jupyter notebook taskD.ipynb   # Course Improvement Recommender
jupyter notebook phase3.ipynb  # Cold start / real-time inferencing demo
```

---

## Regenerating the Dataset (Optional)

If you need to recreate the datasets from scratch, run the notebooks inside `creating_dataset/` in this order:

```
students.ipynb → courses.ipynb → concepts.ipynb → concepts_preq.ipynb
→ course_concepts.ipynb → course_prerequisites.ipynb → enrollments.ipynb
→ assessments.ipynb → chatbots_signals.ipynb → graph_edges.ipynb → embeddings.ipynb
```

> **Note:** `embeddings.ipynb` requires a local embedding model running via Ollama on port 11434 (`mxbai-embed-large`).

---

## Dataset Overview

| File | Rows | Description |
|------|------|-------------|
| students.csv | 2,000 | Student profiles with GPA, effort, modality, risk |
| courses.csv | 40 | Courses across 4 tiers and 5 departments |
| enrollments.csv | 41,985 | Student-course enrollments with grades |
| concepts.csv | 80 | Concepts across 6 domains |
| assessment_scores.csv | 419,850 | 10 assessments per enrollment |
| chatbot_signals.csv | 35,006 | Student-concept chatbot interactions |
| graph_edges.csv | 77,301 | All graph edges unified |

---

## Team

| Person | Task | Role |
|--------|------|------|
| Tushar | Task A — Student Success Navigator | Graph Service |
| Neha | Task B — Concept Sequencing Planner | Demo + Evaluation |
| Deepak | Task C — Learning Modality Match | Graph Construction |
| Matthew | Task D — Course Improvement Recommender | Data & Dataset |