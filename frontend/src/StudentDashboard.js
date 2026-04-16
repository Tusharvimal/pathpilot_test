import { useState, useEffect } from 'react';

const API = 'http://localhost:8000/api';

function StudentDashboard() {
  const [mode, setMode] = useState('existing');
  const [studentList, setStudentList] = useState([]);
  const [selectedStudent, setSelectedStudent] = useState('');
  const [studentData, setStudentData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Cold start state
  const [newGpa, setNewGpa] = useState('');
  const [coldStartData, setColdStartData] = useState(null);

  // Course advice state
  const [advice, setAdvice] = useState(null);
  const [adviceLoading, setAdviceLoading] = useState(false);

  // Learning path state
  const [learningPath, setLearningPath] = useState(null);
  const [pathLoading, setPathLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/students`)
      .then(res => res.json())
      .then(data => setStudentList(data));
  }, []);

  const loadStudent = (studentId) => {
    setSelectedStudent(studentId);
    setStudentData(null);
    setAdvice(null);
    setLearningPath(null);
    if (!studentId) return;

    setLoading(true);
    fetch(`${API}/student/${studentId}`)
      .then(res => res.json())
      .then(data => {
        setStudentData(data);
        setLoading(false);
      });
  };

  const predictColdStart = () => {
    if (!newGpa) return;
    setLoading(true);
    setColdStartData(null);

    fetch(`${API}/predict/cold-start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gpa: parseFloat(newGpa), completed_courses: [] })
    })
      .then(res => res.json())
      .then(data => {
        setColdStartData(data);
        setLoading(false);
      });
  };

  const loadAdvice = (courseId) => {
    if (mode === 'existing' && selectedStudent) {
      setAdviceLoading(true);
      setLearningPath(null);
      fetch(`${API}/student/${selectedStudent}/course-advice/${courseId}`)
        .then(res => res.json())
        .then(data => {
          setAdvice(data);
          setAdviceLoading(false);
        });
    } else {
      setAdviceLoading(true);
      fetch(`${API}/course-report/${courseId}`)
        .then(res => res.json())
        .then(data => {
          setAdvice({ cold_start: true, ...data });
          setAdviceLoading(false);
        });
    }
  };

  const loadLearningPath = (courseId) => {
    if (!selectedStudent) return;
    setPathLoading(true);
    fetch(`${API}/student/${selectedStudent}/learning-path/${courseId}`)
      .then(res => res.json())
      .then(data => {
        setLearningPath(data);
        setPathLoading(false);
      });
  };

  // Modality probability bar component
  const ModalityBar = ({ probs }) => {
    const modalities = [
      { key: 'visual', label: 'Visual', color: '#58a6ff' },
      { key: 'hands_on', label: 'Hands-on', color: '#3fb950' },
      { key: 'analytical', label: 'Analytical', color: '#d29922' },
      { key: 'exploratory', label: 'Exploratory', color: '#bc8cff' },
    ];

    return (
      <div style={{ marginTop: '8px' }}>
        {modalities.map(m => (
          <div key={m.key} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span style={{ width: '90px', fontSize: '12px', color: '#8b949e' }}>{m.label}</span>
            <div style={{ flex: 1, height: '8px', background: '#21262d', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                width: `${(probs[m.key] || 0) * 100}%`,
                height: '100%',
                background: m.color,
                borderRadius: '4px',
                transition: 'width 0.5s'
              }}></div>
            </div>
            <span style={{ width: '45px', fontSize: '12px', color: '#8b949e', textAlign: 'right' }}>
              {((probs[m.key] || 0) * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    );
  };

  // Learning Path panel
  const LearningPathPanel = () => {
    if (pathLoading) return <div className="loading">Loading learning path...</div>;
    if (!learningPath) return null;

    return (
      <div className="card" style={{ borderColor: '#bc8cff' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2>Learning Path: {learningPath.course_name}</h2>
          <button
            onClick={() => setLearningPath(null)}
            style={{ background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', fontSize: '18px' }}
          >✕</button>
        </div>

        {/* Risk badge */}
        <div style={{ marginBottom: '16px' }}>
          <span className={`risk-badge risk-${learningPath.risk.class.toLowerCase()}`}>
            {learningPath.risk.class} Risk
          </span>
          <span style={{ color: '#8b949e', fontSize: '12px', marginLeft: '8px' }}>
            ({(Math.max(learningPath.risk.probs.low, learningPath.risk.probs.medium, learningPath.risk.probs.high) * 100).toFixed(0)}% confidence)
          </span>
        </div>

        {/* Sequenced path */}
        {learningPath.learning_path && learningPath.learning_path.length > 0 ? (
          <div>
            <h3>Recommended Study Sequence</h3>
            <p style={{ color: '#8b949e', fontSize: '12px', marginBottom: '12px' }}>
              Study these concepts in order — foundations first, then your weak areas
            </p>
            <div style={{ position: 'relative', paddingLeft: '24px' }}>
              {/* Vertical line */}
              <div style={{
                position: 'absolute', left: '11px', top: '8px', bottom: '8px',
                width: '2px', background: '#30363d'
              }}></div>

              {learningPath.learning_path.map((step, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'flex-start', gap: '12px',
                  marginBottom: '12px', position: 'relative'
                }}>
                  {/* Circle marker */}
                  <div style={{
                    position: 'absolute', left: '-20px', top: '4px',
                    width: '12px', height: '12px', borderRadius: '50%',
                    background: step.type === 'WEAK' ? '#f85149' : '#30363d',
                    border: `2px solid ${step.type === 'WEAK' ? '#f85149' : '#58a6ff'}`,
                    zIndex: 1
                  }}></div>

                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 600, color: '#e1e4e8' }}>
                        {i + 1}. {step.name}
                      </span>
                      <span style={{
                        fontSize: '11px', padding: '2px 8px', borderRadius: '10px',
                        background: step.type === 'WEAK' ? '#3d1214' : '#1c2d4a',
                        color: step.type === 'WEAK' ? '#f85149' : '#58a6ff',
                        border: `1px solid ${step.type === 'WEAK' ? '#f8514933' : '#58a6ff33'}`
                      }}>
                        {step.type === 'WEAK' ? `Confusion: ${step.confusion.toFixed(3)}` : 'Foundation'}
                      </span>
                    </div>
                    <span style={{ fontSize: '11px', color: '#484f58' }}>{step.concept_id}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p style={{ color: '#3fb950' }}>No concept gaps found — you're ready for this course!</p>
        )}

        {/* Never encountered */}
        {learningPath.never_encountered && learningPath.never_encountered.length > 0 && (
          <div style={{ marginTop: '16px' }}>
            <h3 style={{ color: '#d29922' }}>Never Encountered</h3>
            <p style={{ color: '#8b949e', fontSize: '12px', marginBottom: '8px' }}>
              You have no interaction history with these concepts — consider reviewing them
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {learningPath.never_encountered.map((c, i) => (
                <span key={i} style={{
                  fontSize: '12px', padding: '4px 10px', borderRadius: '12px',
                  background: '#2d2000', color: '#d29922', border: '1px solid #d2992233'
                }}>
                  {c.concept_name || c.name}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Advice panel component
  const AdvicePanel = () => {
    if (adviceLoading) return <div className="loading">Loading advice...</div>;
    if (!advice) return null;

    // Cold start — show general course info
    if (advice.cold_start) {
      return (
        <div className="card" style={{ borderColor: '#58a6ff' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2>Course Info: {advice.course_name}</h2>
            <button
              onClick={() => setAdvice(null)}
              style={{ background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', fontSize: '18px' }}
            >✕</button>
          </div>

          <p style={{ color: '#8b949e', marginBottom: '16px' }}>
            Since you're a new student, here's what other students found challenging in this course.
          </p>

          {/* Systemic gaps if available */}
          {advice.systemic_gaps && advice.systemic_gaps.length > 0 && (
            <div>
              <h3>Common Student Struggles</h3>
              <table className="table">
                <thead>
                  <tr>
                    <th>Concept</th>
                    <th>Students Struggling</th>
                    <th>Avg Confusion</th>
                  </tr>
                </thead>
                <tbody>
                  {advice.systemic_gaps.map((c, i) => (
                    <tr key={i}>
                      <td>{c.concept_name}</td>
                      <td>{c.pct_of_high_risk}%</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{ width: '60px', height: '8px', background: '#21262d', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${c.avg_confusion * 100}%`, height: '100%', background: c.avg_confusion > 0.5 ? '#da3633' : '#d29922', borderRadius: '4px' }}></div>
                          </div>
                          <span>{c.avg_confusion.toFixed(3)}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Fallback to confusing concepts if no systemic gaps */}
          {(!advice.systemic_gaps || advice.systemic_gaps.length === 0) && advice.confusing_concepts && advice.confusing_concepts.length > 0 && (
            <div>
              <h3>Commonly Difficult Concepts</h3>
              <table className="table">
                <thead>
                  <tr>
                    <th>Concept</th>
                    <th>Avg Confusion</th>
                  </tr>
                </thead>
                <tbody>
                  {advice.confusing_concepts.map((c, i) => (
                    <tr key={i}>
                      <td>{c.concept_name}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{ width: '60px', height: '8px', background: '#21262d', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${c.avg_confusion * 100}%`, height: '100%', background: c.avg_confusion > 0.5 ? '#da3633' : '#d29922', borderRadius: '4px' }}></div>
                          </div>
                          <span>{c.avg_confusion.toFixed(3)}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      );
    }

    // Existing student — personalized advice
    return (
      <div className="card" style={{ borderColor: '#58a6ff' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2>Course Advice: {advice.course_name}</h2>
          <button
            onClick={() => setAdvice(null)}
            style={{ background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', fontSize: '18px' }}
          >✕</button>
        </div>

        {/* Risk prediction */}
        {advice.risk && (
          <div style={{ marginBottom: '16px' }}>
            <span className={`risk-badge risk-${advice.risk.class.toLowerCase()}`}>
              {advice.risk.class} Risk
            </span>
            <span style={{ color: '#8b949e', fontSize: '12px', marginLeft: '8px' }}>
              ({(Math.max(advice.risk.probs.low, advice.risk.probs.medium, advice.risk.probs.high) * 100).toFixed(0)}% confidence)
            </span>
          </div>
        )}

        {/* Task C: Modality prediction */}
        {advice.modality && (
          <div style={{
            padding: '12px 16px',
            borderRadius: '6px',
            marginBottom: '16px',
            background: advice.modality.match ? '#0d4429' : '#4a3000',
            border: `1px solid ${advice.modality.match ? '#3fb950' : '#d29922'}`
          }}>
            <div style={{ fontWeight: 600, marginBottom: '4px' }}>
              {advice.modality.match ? '✓ Modality Match' : '⚠ Modality Mismatch'}
            </div>
            <div style={{ fontSize: '13px', color: '#e1e4e8' }}>
              {advice.modality.recommendation}
            </div>
            <div style={{ fontSize: '12px', color: '#8b949e', marginTop: '4px' }}>
              Predicted style: <strong>{advice.modality.predicted}</strong> | Course delivers: <strong>{advice.modality.course}</strong>
            </div>

            {/* Modality probabilities */}
            {advice.modality.probs && (
              <ModalityBar probs={advice.modality.probs} />
            )}
          </div>
        )}

        {/* Task B: Weak Concepts */}
        {advice.weak_concepts && advice.weak_concepts.length > 0 && (
          <div>
            <h3>Your Weak Concepts</h3>
            <p style={{ color: '#8b949e', fontSize: '12px', marginBottom: '8px' }}>
              Identified using GNN prereq analysis + your chatbot history
            </p>
            <table className="table">
              <thead>
                <tr>
                  <th>Concept</th>
                  <th>Confusion Score</th>
                </tr>
              </thead>
              <tbody>
                {advice.weak_concepts.map((c, i) => (
                  <tr key={i}>
                    <td>{c.concept_name}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ width: '60px', height: '8px', background: '#21262d', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{ width: `${c.confusion_score * 100}%`, height: '100%', background: c.confusion_score > 0.5 ? '#da3633' : '#d29922', borderRadius: '4px' }}></div>
                        </div>
                        <span>{c.confusion_score.toFixed(3)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Never encountered concepts */}
        {advice.never_encountered && advice.never_encountered.length > 0 && (
          <div style={{ marginTop: '16px' }}>
            <h3 style={{ color: '#d29922' }}>Never Encountered</h3>
            <p style={{ color: '#8b949e', fontSize: '12px', marginBottom: '8px' }}>
              You have no history with these prerequisite concepts
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {advice.never_encountered.map((c, i) => (
                <span key={i} style={{
                  fontSize: '12px', padding: '4px 10px', borderRadius: '12px',
                  background: '#2d2000', color: '#d29922', border: '1px solid #d2992233'
                }}>
                  {c.concept_name || c.name}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Button to load full learning path */}
        {advice.risk && advice.risk.class === 'High' && (
          <button
            className="btn btn-primary"
            style={{ marginTop: '16px', width: '100%' }}
            onClick={() => loadLearningPath(advice.course_id)}
          >
            Generate Learning Path
          </button>
        )}
      </div>
    );
  };

  return (
    <div>
      {/* Mode toggle */}
      <div className="mode-toggle">
        <button
          className={mode === 'existing' ? 'btn btn-primary' : 'btn'}
          onClick={() => { setMode('existing'); setColdStartData(null); setAdvice(null); setLearningPath(null); }}
          style={mode !== 'existing' ? { background: '#21262d', color: '#8b949e' } : {}}
        >
          Existing Student
        </button>
        <button
          className={mode === 'new' ? 'btn btn-primary' : 'btn'}
          onClick={() => { setMode('new'); setStudentData(null); setAdvice(null); setLearningPath(null); }}
          style={mode !== 'new' ? { background: '#21262d', color: '#8b949e' } : {}}
        >
          New Student (Cold Start)
        </button>
      </div>

      {mode === 'existing' ? (
        <div>
          <div className="card">
            <h2>Select Student</h2>
            <select onChange={(e) => loadStudent(e.target.value)} value={selectedStudent}>
              <option value="">-- Choose a student --</option>
              {studentList.map(s => (
                <option key={s.student_id} value={s.student_id}>
                  {s.student_id} — GPA: {s.gpa} — {s.modality}
                </option>
              ))}
            </select>
          </div>

          {loading && <div className="loading">Loading...</div>}

          {studentData && (
            <>
              <div className="card">
                <h2>Student Profile</h2>
                <div className="profile-stats">
                  <div className="stat">
                    <div className="stat-value">{studentData.student_id}</div>
                    <div className="stat-label">Student ID</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">{studentData.gpa.toFixed(2)}</div>
                    <div className="stat-label">Starting GPA</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">{studentData.modality}</div>
                    <div className="stat-label">Learning Style</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">{studentData.completed_courses.length}</div>
                    <div className="stat-label">Courses Taken</div>
                  </div>
                </div>
              </div>

              {/* Advice panel */}
              <AdvicePanel />

              {/* Learning path panel */}
              <LearningPathPanel />

              <div className="grid-2">
                <div className="card">
                  <h2>Completed Courses</h2>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Course</th>
                        <th>Sem</th>
                        <th>Grade</th>
                        <th>Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {studentData.completed_courses.map((c, i) => (
                        <tr key={i}>
                          <td>{c.course_name}</td>
                          <td>{c.semester}</td>
                          <td>{c.grade.toFixed(2)}</td>
                          <td>
                            <span className={`risk-badge risk-${c.risk_class.toLowerCase()}`}>
                              {c.risk_class}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="card">
                  <h2>Upcoming Course Predictions</h2>
                  <p style={{ color: '#8b949e', fontSize: '12px', marginBottom: '8px' }}>
                    Click any course for detailed advice
                  </p>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Course</th>
                        <th>Tier</th>
                        <th>Risk</th>
                        <th>Probability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {studentData.upcoming_courses.map((c, i) => (
                        <tr
                          key={i}
                          onClick={() => loadAdvice(c.course_id)}
                          style={{ cursor: 'pointer' }}
                        >
                          <td>{c.course_name}</td>
                          <td>{c.tier}</td>
                          <td>
                            <span className={`risk-badge risk-${c.risk_class.toLowerCase()}`}>
                              {c.risk_class}
                            </span>
                            <span style={{ color: '#8b949e', fontSize: '12px', marginLeft: '8px' }}>
                              {(Math.max(c.probs.low, c.probs.medium, c.probs.high) * 100).toFixed(0)}%
                            </span>
                          </td>
                          <td>
                            <div className="prob-bar">
                              <div className="prob-low" style={{ width: `${c.probs.low * 100}%` }}></div>
                              <div className="prob-medium" style={{ width: `${c.probs.medium * 100}%` }}></div>
                              <div className="prob-high" style={{ width: `${c.probs.high * 100}%` }}></div>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      ) : (
        <div>
          <div className="card">
            <h2>New Student Registration</h2>
            <p style={{ color: '#8b949e', marginBottom: '12px' }}>
              Enter your GPA to get initial course risk predictions
            </p>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              <input
                type="number"
                step="0.1"
                min="0"
                max="4"
                placeholder="Enter GPA (e.g. 3.2)"
                value={newGpa}
                onChange={(e) => setNewGpa(e.target.value)}
                style={{ maxWidth: '200px' }}
              />
              <button className="btn btn-primary" onClick={predictColdStart}>
                Get Predictions
              </button>
            </div>
          </div>

          {loading && <div className="loading">Loading predictions...</div>}

          {coldStartData && (
            <>
              <AdvicePanel />

              <div className="card">
                <h2>Risk Predictions (GPA: {coldStartData.gpa})</h2>
                <p style={{ color: '#8b949e', marginBottom: '12px' }}>
                  Based on similar students (KNN) — Alpha: {coldStartData.alpha}
                </p>
                <p style={{ color: '#8b949e', fontSize: '12px', marginBottom: '8px' }}>
                  Click any course for more details
                </p>
                <table className="table">
                  <thead>
                    <tr>
                      <th>Course</th>
                      <th>Tier</th>
                      <th>Dept</th>
                      <th>Risk</th>
                      <th>Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {coldStartData.predictions.map((c, i) => (
                      <tr
                        key={i}
                        onClick={() => loadAdvice(c.course_id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td>{c.course_name}</td>
                        <td>{c.tier}</td>
                        <td>{c.department}</td>
                        <td>
                          <span className={`risk-badge risk-${c.risk_class.toLowerCase()}`}>
                            {c.risk_class}
                          </span>
                          <span style={{ color: '#8b949e', fontSize: '12px', marginLeft: '8px' }}>
                            {(Math.max(c.probs.low, c.probs.medium, c.probs.high) * 100).toFixed(0)}%
                          </span>
                        </td>
                        <td>
                          <div className="prob-bar">
                            <div className="prob-low" style={{ width: `${c.probs.low * 100}%` }}></div>
                            <div className="prob-medium" style={{ width: `${c.probs.medium * 100}%` }}></div>
                            <div className="prob-high" style={{ width: `${c.probs.high * 100}%` }}></div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default StudentDashboard;