from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import torch.nn as nn
from model import GraphSAGE, LinkPredictor
from utils import load_data, normalize_student_features, add_cold_start_student, get_knn_prior, TaskBPipeline

app = FastAPI(title = "PathPilot API")

app.add_middleware(
    CORSMiddleware, 
    allow_origins = ['http://localhost:3000'],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

checkpoint = torch.load('models/task_a_model_new.pt',weights_only=False, map_location='cpu')
data = checkpoint['data']
student_id_map = checkpoint['student_id_map']
course_id_map = checkpoint['course_id_map']
concept_id_map = checkpoint['concept_id_map']

model = GraphSAGE(hidden_dim=256)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

link_model = LinkPredictor(emb_dim=256)
link_model.load_state_dict(torch.load('models/task_b_link_model_new.pt', weights_only=True, map_location='cpu'))
link_model.eval()

with torch.no_grad():
    out = model(data)
    concept_embs = out[2]

idx_to_concept = {v: k for k, v in concept_id_map.items()}

db = load_data()
students = db['students']
courses = db['courses']
enrollments = db['enrollments']
chatbot = db['chatbot']
concepts = db['concepts']
course_concepts = db['course_concepts']
raw_means = db['raw_means']
raw_stds = db['raw_stds']
course_name_map = db['course_name_map']
concept_name_map = db['concept_name_map']

pipeline = TaskBPipeline(model, link_model, data, concept_id_map,
                         idx_to_concept, concept_embs, chatbot,
                         course_concepts, concept_name_map)

# Load Task C modality model
modality_model = nn.Sequential(
    nn.Linear(260, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 4)
)
modality_model.load_state_dict(torch.load('models/task_c_model_new.pt', weights_only=True, map_location='cpu'))
modality_model.eval()

# Precompute grade-by-modality features for all students
course_modality_map = courses.set_index('course_id')['best_modality'].to_dict()
enrollments['course_modality'] = enrollments['course_id'].map(course_modality_map)
grade_by_modality = enrollments.groupby(['student_id', 'course_modality'])['grade'].mean().unstack(fill_value=0)
overall_avg = grade_by_modality.mean(axis=1)
for col in ['analytical', 'exploratory', 'hands_on', 'visual']:
    grade_by_modality[col] = grade_by_modality[col] - overall_avg
grade_by_modality = grade_by_modality.reindex(students['student_id'])
modality_features = torch.tensor(
    grade_by_modality[['analytical', 'exploratory', 'hands_on', 'visual']].values,
    dtype=torch.float
)

modality_labels = ['visual', 'hands_on', 'analytical', 'exploratory']

print("Task C modality model loaded!")

def predict_risk(s_emb, course_embs, cid):
    c_emb = course_embs[course_id_map[cid]]
    pair = torch.cat([s_emb, c_emb, s_emb - c_emb, s_emb * c_emb]).unsqueeze(0)
    logits = model.task_a_head(pair)
    probs = torch.softmax(logits, dim = 1).squeeze().tolist()
    pred = ['Low', 'Medium', 'High'][logits.argmax().item()]
    return pred, probs

@app.get('/api/health')
def health():
    return {
        "status": "ok", 
        "students": len(students),
        "courses": len(courses),
        "model": "GraphSAGE-256"
    }

@app.get('/api/students/')
def list_students():
    demo_students = students[students['split'].isin(['val', 'test'])]
    result = []
    for _, row in demo_students.iterrows():
        result.append({
            "student_id": row['student_id'],
            "gpa": round(float(row['gpa_start']),2),
            "modality": row['true_modality']
        })
    return result

@app.get("/api/courses")
def list_courses():
    result = []
    for _, row in courses.iterrows():
        result.append({
            "course_id": row['course_id'],
            "course_name": row['course_name'],
            "tier": int(row['tier']),
            "department": row['department'],
            "difficulty": float(row['difficulty']),
        })
    return result

@app.get("/api/student/{student_id}")
def get_student(student_id):
    student = students[students['student_id'] == student_id]
    if len(student) == 0:
        return {"error": "Student not found"}
    student = student.iloc[0]

    student_enr = enrollments[enrollments['student_id'] == student_id].sort_values('semester')
    completed = []
    for _, row in student_enr.iterrows():
        completed.append({
            "course_id": row['course_id'],
            "course_name": course_name_map[row['course_id']],
            "semester": int(row['semester']),
            "grade": float(row['grade']),
            "risk_class": row['risk_class'],
        })

    taken_ids = set(student_enr['course_id'].tolist())
    not_taken = set(courses['course_id'].tolist()) - taken_ids

    s_idx = student_id_map[student_id]
    upcoming = []

    with torch.no_grad():
        out = model(data)
        s_emb = out[0][s_idx]

        for cid in sorted(not_taken):
            pred, probs = predict_risk(s_emb, out[1], cid)
            course_info = courses[courses['course_id'] == cid].iloc[0]
            upcoming.append({
                "course_id": cid,
                "course_name": course_name_map[cid],
                "tier": int(course_info['tier']),
                "department": course_info['department'],
                "risk_class": pred,
                "probs": {"low": round(probs[0], 3), "medium": round(probs[1], 3), "high": round(probs[2], 3)},
            })

    return {
        "student_id": student_id,
        "gpa": float(student['gpa_start']),
        "modality": student['true_modality'],
        "completed_courses": completed,
        "upcoming_courses": sorted(upcoming, key=lambda x: x['tier']),
    }

class ColdStartRequest(BaseModel):
    gpa: float
    effort: Optional[float] = None
    risk_factor: Optional[float] = None
    avg_confusion: Optional[float] = 0.0
    avg_questions: Optional[float] = 0.0
    completed_courses: Optional[list] = []

@app.post("/api/predict/cold-start")
def cold_start_predict(req: ColdStartRequest):
    features = normalize_student_features(
        req.gpa, raw_means, raw_stds,
        effort = req.effort, risk = req.risk_factor, 
        confusion = req.avg_confusion, questions = req.avg_questions
    )

    sim, new_idx = add_cold_start_student(data, features, req.completed_courses, course_id_map)
    n_edges = len(req.completed_courses)
    completed_set = set(req.completed_courses)
    not_taken = set(courses['course_id'].tolist()) - completed_set

    predictions = []
    with torch.no_grad():
        out = model(sim)
        s_emb = out[0][new_idx]

        for cid in sorted(not_taken):
            knn = get_knn_prior(req.gpa, cid, students, enrollments)
            _, model_probs = predict_risk(s_emb, out[1], cid)

            if n_edges == 0:
                alpha = 1.0
            elif n_edges <= 10:
                alpha = 0.6
            else:
                alpha = 0.1

            blended = [alpha * k + (1-alpha)*m for k, m in zip(knn, model_probs)]
            total = sum(blended)
            blended = [b/total for b in blended]
            pred = ['Low', 'Medium', 'High'][blended.index(max(blended))]

            course_info = courses[courses['course_id'] == cid].iloc[0]
            predictions.append({
                "course_id": cid,
                "course_name": course_name_map[cid],
                "tier": int(course_info['tier']),
                "department": course_info['department'],
                "risk_class": pred,
                "probs": {"low": round(blended[0], 3), "medium": round(blended[1], 3), "high": round(blended[2], 3)},
            })

    return {
        "gpa": req.gpa,
        "n_edges": n_edges,
        "alpha": alpha if n_edges > 0 else 1.0,
        "predictions": sorted(predictions, key=lambda x: x['tier']),
    }

@app.get('/api/student/{student_id}/course-advice/{course_id}')
def course_advice(student_id: str, course_id: str):
    student = students[students['student_id'] == student_id]
    if len(student) == 0:
        return {"error": "Student not found"}
    student = student.iloc[0]

    course_info = courses[courses['course_id'] == course_id].iloc[0]
    course_modality = course_info['best_modality']

    # Task C: predict student's modality using the model
    s_idx = student_id_map[student_id]
    with torch.no_grad():
        out = model(data)
        s_emb = out[0][s_idx]
        combined = torch.cat([s_emb, modality_features[s_idx]]).unsqueeze(0)
        modality_preds = modality_model(combined)
        modality_probs = torch.softmax(modality_preds, dim=1).squeeze().tolist()
        predicted_modality = modality_labels[modality_preds.argmax().item()]

    modality_match = predicted_modality == course_modality

    # Task B: weak concepts
    weak, no_data = pipeline.get_weak_concepts(student_id, course_id, top_k=5)
    
    weak_concepts = []
    for concept_id, confusion in weak:
        weak_concepts.append({
            "concept_id": concept_id,
            "concept_name": concept_name_map.get(concept_id, concept_id),
            "confusion_score": round(float(confusion), 3),
        })

    # Task A: risk prediction
    pred, probs = predict_risk(s_emb, out[1], course_id)

    return {
        "student_id": student_id,
        "course_id": course_id,
        "course_name": course_name_map[course_id],
        "risk": {
            "class": pred,
            "probs": {"low": round(probs[0], 3), "medium": round(probs[1], 3), "high": round(probs[2], 3)}
        },
        "modality": {
            "predicted": predicted_modality,
            "probs": {
                "visual": round(modality_probs[0], 3),
                "hands_on": round(modality_probs[1], 3),
                "analytical": round(modality_probs[2], 3),
                "exploratory": round(modality_probs[3], 3)
            },
            "course": course_modality,
            "match": modality_match,
            "recommendation": f"Your predicted learning style is {predicted_modality}. This course uses {course_modality}."
                              + (" Great match!" if modality_match else f" Consider supplementing with {predicted_modality}-based resources.")
        },
        "weak_concepts": weak_concepts,
        "never_encountered": [{"concept_id": c, "concept_name": concept_name_map.get(c, c)} for c, _ in no_data],
    }

@app.get('/api/student/{student_id}/learning-path/{course_id}')
def learning_path(student_id: str, course_id: str):
    if student_id not in student_id_map:
        return {"error": "Student not found"}
    if course_id not in course_id_map:
        return {"error": "Course not found"}
    
    # Get Task A risk prediction
    s_idx = student_id_map[student_id]
    with torch.no_grad():
        out = model(data)
        s_emb = out[0][s_idx]
        pred, probs = predict_risk(s_emb, out[1], course_id)
    
    # Get Task B learning path
    result = pipeline.get_learning_path(student_id, course_id, top_k=3)
    
    return {
        "student_id": student_id,
        "course_id": course_id,
        "course_name": course_name_map[course_id],
        "risk": {
            "class": pred,
            "probs": {"low": round(probs[0], 3), "medium": round(probs[1], 3), "high": round(probs[2], 3)}
        },
        "learning_path": result['path'],
        "weak_concepts": result['weak'],
        "never_encountered": result['no_data']
    }

@app.get('/api/course-report/{course_id}')
def course_report(course_id: str):
    if course_id not in course_name_map:
        return {"error": "Course not found"}
    
    course_info = courses[courses['course_id'] == course_id].iloc[0]

    # Basic stats
    course_enr = enrollments[enrollments['course_id'] == course_id]
    total = len(course_enr)
    risk_counts = course_enr['risk_class'].value_counts().to_dict()
    avg_grade = float(course_enr['grade'].mean())
    fail_rate = float((course_enr['passed'] == 0).mean())

    # Systemic gaps using Task B pipeline
    high_risk_students = course_enr[
        course_enr['risk_class'] == 'High'
    ]['student_id'].tolist()

    systemic_gaps = []
    if high_risk_students:
        concept_counts = {}
        concept_total_confusion = {}

        for student_id in high_risk_students:
            weak, no_data = pipeline.get_weak_concepts(student_id, course_id, top_k=3)
            for cid, confusion in weak:
                if cid not in concept_counts:
                    concept_counts[cid] = 0
                    concept_total_confusion[cid] = 0
                concept_counts[cid] += 1
                concept_total_confusion[cid] += confusion

        for cid, count in concept_counts.items():
            avg_conf = concept_total_confusion[cid] / count
            systemic_gaps.append({
                "concept_id": cid,
                "concept_name": concept_name_map.get(cid, cid),
                "student_count": count,
                "pct_of_high_risk": round(count / len(high_risk_students) * 100, 1),
                "avg_confusion": round(avg_conf, 3)
            })

        systemic_gaps.sort(key=lambda x: x['student_count'], reverse=True)
        systemic_gaps = systemic_gaps[:5]

    # Simulation: what if we add tutoring?
    simulation_results = []
    covered = course_concepts[course_concepts['course_id'] == course_id]['concept_id'].tolist()
    
    student_indices = torch.tensor([student_id_map[s] for s in enrollments['student_id']], dtype=torch.long)
    course_indices = torch.tensor([course_id_map[c] for c in enrollments['course_id']], dtype=torch.long)
    course_mask = (enrollments['course_id'] == course_id).values

    with torch.no_grad():
        out = model(data)
        pairs = torch.cat([out[0][student_indices], out[1][course_indices],
                           out[0][student_indices] - out[1][course_indices],
                           out[0][student_indices] * out[1][course_indices]], dim=1)
        baseline_preds = model.task_a_head(pairs).argmax(dim=1)

    baseline_high = int((baseline_preds[course_mask] == 2).sum().item())

    confusion_lookup = chatbot.set_index(['student_id', 'concept_id'])['confusion_score'].to_dict()
    course_students = course_enr['student_id'].tolist()

    for concept_id in covered:
        affected = []
        for sid in course_students:
            if (sid, concept_id) in confusion_lookup:
                affected.append(student_id_map[sid])

        if not affected:
            continue

        original_x = data['student'].x.clone()

        for s_idx in affected:
            data['student'].x[s_idx, 3] = data['student'].x[s_idx, 3] * 0.5

        with torch.no_grad():
            out = model(data)
            pairs = torch.cat([out[0][student_indices], out[1][course_indices],
                               out[0][student_indices] - out[1][course_indices],
                               out[0][student_indices] * out[1][course_indices]], dim=1)
            new_preds = model.task_a_head(pairs).argmax(dim=1)

        new_high = int((new_preds[course_mask] == 2).sum().item())
        helped = baseline_high - new_high

        simulation_results.append({
            "concept_id": concept_id,
            "concept_name": concept_name_map.get(concept_id, concept_id),
            "students_affected": len(affected),
            "students_helped": helped
        })

        data['student'].x = original_x

    simulation_results.sort(key=lambda x: x['students_helped'], reverse=True)
    simulation_results = simulation_results[:5]

    return {
        "course_id": course_id,
        "course_name": course_name_map[course_id],
        "tier": int(course_info['tier']),
        "department": course_info['department'],
        "difficulty": float(course_info['difficulty']),
        "total_enrolled": total,
        "avg_grade": round(avg_grade, 2),
        "fail_rate": round(fail_rate, 3),
        "risk_distribution": {
            "low": risk_counts.get('Low', 0),
            "medium": risk_counts.get('Medium', 0),
            "high": risk_counts.get('High', 0),
        },
        "systemic_gaps": systemic_gaps,
        "simulation": {
            "baseline_high_risk": baseline_high,
            "interventions": simulation_results
        }
    }