import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv

## TSNE

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv1 = nn.ModuleDict({
            'enrolled_in': GATConv((-1, -1), hidden_dim, edge_dim=1, add_self_loops=False),
            'engaged_with': GATConv((-1, -1), hidden_dim, edge_dim=1, add_self_loops=False),
            'prereq_of': SAGEConv(-1, hidden_dim),
            'covers': SAGEConv(-1, hidden_dim),
            'concept_prereq': SAGEConv(-1, hidden_dim),
            'rev_enrolled_in': GATConv((-1, -1), hidden_dim, edge_dim=1, add_self_loops=False),
            'rev_engaged_with': GATConv((-1, -1), hidden_dim, edge_dim=1, add_self_loops=False)
        })

        self.conv2 = nn.ModuleDict({
            'rev_enrolled_in': SAGEConv(hidden_dim, hidden_dim),
            'rev_engaged_with': SAGEConv(hidden_dim, hidden_dim)
        })

        self.dropout = nn.Dropout(0.3)
        self.student_proj = nn.Linear(5, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.task_a_head = nn.Linear(hidden_dim*4, 3)
        # self.task_a_head = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.3), 
        #     nn.Linear(hidden_dim, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32, 3)
        # )

    def forward(self, data):

        x_student = data['student'].x
        x_course = data['course'].x
        x_concept = data['concept'].x

        ## Layer 1 
        course_from_students = self.conv1['enrolled_in'](
            (x_student, x_course),
            data['student', 'enrolled_in', 'course'].edge_index,
            edge_attr=data['student', 'enrolled_in', 'course'].edge_weight.unsqueeze(1)
        )
        course_from_course = self.conv1['prereq_of'](
            (x_course, x_course),
            data['course', 'prereq_of', 'course'].edge_index
        )
        concepts_from_students = self.conv1['engaged_with'](
            (x_student, x_concept),
            data['student', 'engaged_with', 'concept'].edge_index,
            edge_attr=data['student', 'engaged_with', 'concept'].edge_weight.unsqueeze(1)
        )
        concepts_from_course = self.conv1['covers'](
            (x_course, x_concept),
            data['course', 'covers', 'concept'].edge_index
        )
        concepts_from_concepts = self.conv1['concept_prereq'](
            (x_concept, x_concept),
            data['concept', 'concept_prereq', 'concept'].edge_index
        )
        student_from_course_l1 = self.conv1['rev_enrolled_in'](
            (x_course, x_student),
            data['course', 'rev_enrolled_in', 'student'].edge_index,
            edge_attr=data['course', 'rev_enrolled_in', 'student'].edge_weight.unsqueeze(1)
        )
        student_from_concept_l1 = self.conv1['rev_engaged_with'](
            (x_concept, x_student),
            data['concept', 'rev_engaged_with', 'student'].edge_index,
            edge_attr=data['concept', 'rev_engaged_with', 'student'].edge_weight.unsqueeze(1)
        )

        # Relu
        x_student_skip = self.student_proj(x_student) * 0.3
        x_course_1 = self.dropout(torch.relu(course_from_students + course_from_course))
        x_concept_1 = self.dropout(torch.relu(concepts_from_students + concepts_from_course + concepts_from_concepts))
        x_student_1 = self.layer_norm(student_from_course_l1 + student_from_concept_l1 + x_student_skip)
        x_student_1 = self.dropout(torch.relu(x_student_1))


        ## Layer 2
        student_from_course = self.conv2['rev_enrolled_in'](
            (x_course_1, x_student_1),
            data['course', 'rev_enrolled_in', 'student'].edge_index,
            # size=(x_course_1.size(0), x_student.size(0))   # ← (num_src, num_dst)
        )

        student_from_concept = self.conv2['rev_engaged_with'](
            (x_concept_1, x_student_1),
            data['concept', 'rev_engaged_with', 'student'].edge_index,
            # size=(x_concept_1.size(0), x_student.size(0))   # ← (num_src, num_dst)
        )

        x_student_1 = self.dropout(torch.relu(student_from_course + student_from_concept))

        return x_student_1, x_course_1, x_concept_1
    

class LinkPredictor(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, src_emb, dst_emb):
        pair = torch.concat([src_emb, dst_emb, src_emb - dst_emb, src_emb * dst_emb], dim = 1)
        return self.net(pair).squeeze()

# link_model = LinkPredictor(emb_dim=256)
# link_model.load_state_dict(torch.load('models/task_b_link_model.pt', weights_only=True))
# link_model.eval()