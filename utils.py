import torch
import copy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque


def load_data():
    """Load all CSVs and compute normalization stats."""
    students = pd.read_csv('datasets/students.csv')
    courses = pd.read_csv('datasets/courses.csv')
    enrollments = pd.read_csv('datasets/enrollments.csv')
    chatbot = pd.read_csv('datasets/chatbot_signals.csv')
    concepts = pd.read_csv('datasets/concepts.csv')
    course_concepts = pd.read_csv('datasets/course_concepts.csv')

    raw_means = {
        'gpa_start': students['gpa_start'].mean(),
        'effort_level': students['effort_level'].mean(),
        'risk_factor': students['risk_factor'].mean(),
        'avg_confusion': chatbot.groupby('student_id')['confusion_score'].mean().mean(),
        'avg_questions': chatbot.groupby('student_id')['num_questions'].mean().mean()
    }

    raw_stds = {
        'gpa_start': students['gpa_start'].std(),
        'effort_level': students['effort_level'].std(),
        'risk_factor': students['risk_factor'].std(),
        'avg_confusion': chatbot.groupby('student_id')['confusion_score'].mean().std(),
        'avg_questions': chatbot.groupby('student_id')['num_questions'].mean().std()
    }

    course_name_map = courses.set_index('course_id')['course_name'].to_dict()
    concept_name_map = concepts.set_index('concept_id')['concept_name'].to_dict()

    return {
        'students': students,
        'courses': courses,
        'enrollments': enrollments,
        'chatbot': chatbot,
        'concepts': concepts,
        'course_concepts': course_concepts,
        'raw_means': raw_means,
        'raw_stds': raw_stds,
        'course_name_map': course_name_map,
        'concept_name_map': concept_name_map,
    }


def normalize_student_features(gpa, raw_means, raw_stds, effort=None, risk=None, confusion=0.0, questions=0.0):
    """Normalize raw student features for the model."""
    if effort is None:
        effort = raw_means['effort_level']
    if risk is None:
        risk = raw_means['risk_factor']

    return torch.tensor([[
        (gpa - raw_means['gpa_start']) / raw_stds['gpa_start'],
        (effort - raw_means['effort_level']) / raw_stds['effort_level'],
        (risk - raw_means['risk_factor']) / raw_stds['risk_factor'],
        (confusion - raw_means['avg_confusion']) / raw_stds['avg_confusion'],
        (questions - raw_means['avg_questions']) / raw_stds['avg_questions']
    ]], dtype=torch.float)


def get_knn_prior(gpa, course_id, students, enrollments, n_neighbors=50):
    """Get risk distribution from similar students for a specific course."""
    students = students.copy()
    students['gpa_diff'] = abs(students['gpa_start'] - gpa)
    similar = students.nsmallest(n_neighbors, 'gpa_diff')['student_id']

    similar_enr = enrollments[
        (enrollments['student_id'].isin(similar)) &
        (enrollments['course_id'] == course_id)
    ]

    if len(similar_enr) == 0:
        return [0.33, 0.34, 0.33]

    risk_dist = similar_enr['risk_class'].value_counts(normalize=True)
    return [
        float(risk_dist.get('Low', 0.0)),
        float(risk_dist.get('Medium', 0.0)),
        float(risk_dist.get('High', 0.0))
    ]


def add_cold_start_student(data, student_features, course_ids, course_id_map):
    """Add a new student node with enrolled_in edges for completed courses."""
    sim = copy.deepcopy(data)
    sim['student'].x = torch.cat([sim['student'].x, student_features], dim=0)
    new_idx = sim['student'].x.shape[0] - 1

    for course_id in course_ids:
        c_idx = course_id_map[course_id]
        new_src = torch.tensor([new_idx], dtype=torch.long)
        new_dst = torch.tensor([c_idx], dtype=torch.long)
        new_weight = torch.tensor([1.0], dtype=torch.float)

        sim['student', 'enrolled_in', 'course'].edge_index = torch.cat([
            sim['student', 'enrolled_in', 'course'].edge_index,
            torch.stack([new_src, new_dst])
        ], dim=1)
        sim['student', 'enrolled_in', 'course'].edge_weight = torch.cat([
            sim['student', 'enrolled_in', 'course'].edge_weight, new_weight
        ])
        sim['course', 'rev_enrolled_in', 'student'].edge_index = torch.cat([
            sim['course', 'rev_enrolled_in', 'student'].edge_index,
            torch.stack([new_dst, new_src])
        ], dim=1)
        sim['course', 'rev_enrolled_in', 'student'].edge_weight = torch.cat([
            sim['course', 'rev_enrolled_in', 'student'].edge_weight, new_weight
        ])

    return sim, new_idx


def predict_blended(model, sim_data, student_idx, course_id, n_edges, gpa, course_id_map, students, enrollments):
    """Blend KNN prior with model prediction based on edge count."""
    knn_prior = get_knn_prior(gpa, course_id, students, enrollments)

    with torch.no_grad():
        out = model(sim_data)
        s_emb = out[0][student_idx]
        c_emb = out[1][course_id_map[course_id]]
        pair = torch.cat([s_emb, c_emb, s_emb - c_emb, s_emb * c_emb]).unsqueeze(0)
        logits = model.task_a_head(pair)
        model_probs = torch.softmax(logits, dim=1).squeeze().tolist()

    if n_edges == 0:
        alpha = 1.0
    elif n_edges <= 10:
        alpha = 0.6
    else:
        alpha = 0.1

    blended = [alpha * k + (1 - alpha) * m for k, m in zip(knn_prior, model_probs)]
    total = sum(blended)
    blended = [b / total for b in blended]

    pred = ['Low', 'Medium', 'High'][blended.index(max(blended))]

    return pred, blended, knn_prior, model_probs


class TaskBPipeline:
    def __init__(self, model, link_model, data, concept_id_map, idx_to_concept,
                 concept_embs, chatbot, course_concepts_df, concept_name_map):
        self.model = model
        self.link_model = link_model
        self.data = data
        self.concept_id_map = concept_id_map
        self.idx_to_concept = idx_to_concept
        self.concept_embs = concept_embs
        self.chatbot = chatbot
        self.course_concepts_df = course_concepts_df
        self.concept_name_map = concept_name_map
    
    def get_prereq_chain(self, concept_ids, top_n=3):
        all_needed = set(concept_ids)
        
        self.link_model.eval()
        with torch.no_grad():
            for concept_id in concept_ids:
                c_idx = self.concept_id_map[concept_id]
                src_emb = self.concept_embs[c_idx].unsqueeze(0).repeat(80, 1)
                
                scores = torch.sigmoid(self.link_model(src_emb, self.concept_embs))
                scores[c_idx] = 0.0
                
                top_indices = scores.argsort(descending=True)[:top_n]
                
                for idx in top_indices:
                    prereq_id = self.idx_to_concept[idx.item()]
                    all_needed.add(prereq_id)
        
        return all_needed
    
    def get_weak_concepts(self, student_id, course_id, top_k=3):
        course_concepts = self.course_concepts_df[
            self.course_concepts_df['course_id'] == course_id
        ]['concept_id'].tolist()
        
        all_needed = self.get_prereq_chain(course_concepts)
        
        student_signals = self.chatbot[self.chatbot['student_id'] == student_id]
        student_confusion = student_signals.set_index('concept_id')['confusion_score'].to_dict()
        
        gap_scores = []
        no_data = []
        for concept_id in all_needed:
            confusion = student_confusion.get(concept_id, None)
            if confusion is not None:
                gap_scores.append((concept_id, confusion))
            else:
                no_data.append((concept_id, -1))
        
        gap_scores.sort(key=lambda x: x[1], reverse=True)
        return gap_scores[:top_k], no_data
    
    def sequence_concepts(self, concept_ids):
        concept_list = list(concept_ids)
        concept_set = set(concept_ids)
        
        pair_scores = {}
        self.link_model.eval()
        with torch.no_grad():
            for concept_id in concept_list:
                c_idx = self.concept_id_map[concept_id]
                src_emb = self.concept_embs[c_idx].unsqueeze(0).repeat(80, 1)
                scores = torch.sigmoid(self.link_model(src_emb, self.concept_embs))
                
                for other_id in concept_list:
                    if other_id == concept_id:
                        continue
                    o_idx = self.concept_id_map[other_id]
                    pair_scores[(concept_id, other_id)] = scores[o_idx].item()
        
        in_degree = {c: 0 for c in concept_set}
        adj = {c: [] for c in concept_set}
        
        for (concept_id, other_id), score in pair_scores.items():
            if score >= 0.7:
                adj[other_id].append(concept_id)
                in_degree[concept_id] += 1
        
        queue = deque([c for c in concept_set if in_degree[c] == 0])
        ordered = []
        
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for child in adj[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        if len(ordered) != len(concept_set):
            remaining = concept_set - set(ordered)
            ordered.extend(sorted(remaining))
        
        return ordered
    
    def get_learning_path(self, student_id, course_id, top_k=3):
        weak, no_data = self.get_weak_concepts(student_id, course_id, top_k)
        
        if not weak:
            return {'weak': [], 'path': [], 'no_data': []}
        
        weak_ids = [c for c, _ in weak]
        weak_scores = {c: s for c, s in weak}
        
        all_needed = self.get_prereq_chain(weak_ids)
        ordered = self.sequence_concepts(all_needed)
        
        path = []
        for concept_id in ordered:
            name = self.concept_name_map.get(concept_id, concept_id)
            if concept_id in weak_scores:
                path.append({
                    'concept_id': concept_id,
                    'name': name,
                    'type': 'WEAK',
                    'confusion': weak_scores[concept_id]
                })
            else:
                path.append({
                    'concept_id': concept_id,
                    'name': name,
                    'type': 'PREREQ',
                    'confusion': None
                })
        
        no_data_list = [{'concept_id': c, 'name': self.concept_name_map.get(c, c)} for c, _ in no_data]
        
        return {
            'weak': [{'concept_id': c, 'confusion': s} for c, s in weak],
            'path': path,
            'no_data': no_data_list
        }