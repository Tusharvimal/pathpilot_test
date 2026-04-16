import { useState, useEffect } from 'react';

const API = 'http://localhost:8000/api';

function ProfessorDashboard() {
  const [courseList, setCourseList] = useState([]);
  const [selectedCourse, setSelectedCourse] = useState('');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/courses`)
      .then(res => res.json())
      .then(data => setCourseList(data));
  }, []);

  const loadReport = (courseId) => {
    setSelectedCourse(courseId);
    setReport(null);
    if (!courseId) return;

    setLoading(true);
    fetch(`${API}/course-report/${courseId}`)
      .then(res => res.json())
      .then(data => {
        setReport(data);
        setLoading(false);
      });
  };

  return (
    <div>
      <div className="card">
        <h2>Select Course</h2>
        <select onChange={(e) => loadReport(e.target.value)} value={selectedCourse}>
          <option value="">-- Choose a course --</option>
          {courseList.map(c => (
            <option key={c.course_id} value={c.course_id}>
              {c.course_id} — {c.course_name} (Tier {c.tier}, {c.department})
            </option>
          ))}
        </select>
      </div>

      {loading && <div className="loading">Loading report...</div>}

      {report && (
        <>
          {/* Course Overview */}
          <div className="card">
            <h2>{report.course_name}</h2>
            <div className="profile-stats">
              <div className="stat">
                <div className="stat-value">Tier {report.tier}</div>
                <div className="stat-label">Level</div>
              </div>
              <div className="stat">
                <div className="stat-value">{report.department}</div>
                <div className="stat-label">Department</div>
              </div>
              <div className="stat">
                <div className="stat-value">{report.difficulty.toFixed(2)}</div>
                <div className="stat-label">Difficulty</div>
              </div>
              <div className="stat">
                <div className="stat-value">{report.total_enrolled}</div>
                <div className="stat-label">Total Enrolled</div>
              </div>
              <div className="stat">
                <div className="stat-value">{report.avg_grade.toFixed(2)}</div>
                <div className="stat-label">Avg Grade</div>
              </div>
              <div className="stat">
                <div className="stat-value">{(report.fail_rate * 100).toFixed(1)}%</div>
                <div className="stat-label">Fail Rate</div>
              </div>
            </div>
          </div>

          <div className="grid-2">
            {/* Risk Distribution */}
            <div className="card">
              <h2>Student Risk Distribution</h2>
              <p style={{ color: '#8b949e', fontSize: '13px', marginBottom: '12px' }}>
                Predicted by GraphSAGE model (Task A)
              </p>
              <div style={{ marginTop: '12px' }}>
                {['low', 'medium', 'high'].map(level => {
                  const count = report.risk_distribution[level];
                  const total = report.total_enrolled;
                  const pct = ((count / total) * 100).toFixed(1);
                  const colors = { low: '#3fb950', medium: '#d29922', high: '#f85149' };

                  return (
                    <div key={level} style={{ marginBottom: '14px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                        <span style={{ textTransform: 'capitalize', fontSize: '15px', color: '#e1e4e8' }}>{level}</span>
                        <span style={{ color: '#c9d1d9', fontSize: '14px' }}>{count} students ({pct}%)</span>
                      </div>
                      <div style={{ height: '14px', background: '#21262d', borderRadius: '7px', overflow: 'hidden' }}>
                        <div style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: colors[level],
                          borderRadius: '7px',
                          transition: 'width 0.5s'
                        }}></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Systemic Gaps — Task B */}
            <div className="card">
              <h2>Systemic Concept Gaps</h2>
              <p style={{ color: '#8b949e', fontSize: '13px', marginBottom: '12px' }}>
                Most common weak concepts across high-risk students (Task B pipeline)
              </p>
              {report.systemic_gaps && report.systemic_gaps.length > 0 ? (
                <table className="table">
                  <thead>
                    <tr>
                      <th>Concept</th>
                      <th>Students</th>
                      <th>Avg Confusion</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.systemic_gaps.map((c, i) => (
                      <tr key={i}>
                        <td style={{ fontSize: '14px' }}>{c.concept_name}</td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontWeight: 600, fontSize: '14px' }}>{c.student_count}</span>
                            <span style={{ color: '#8b949e', fontSize: '13px' }}>
                              ({c.pct_of_high_risk}%)
                            </span>
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <div style={{
                              width: '60px', height: '8px',
                              background: '#21262d', borderRadius: '4px', overflow: 'hidden'
                            }}>
                              <div style={{
                                width: `${c.avg_confusion * 100}%`,
                                height: '100%',
                                background: c.avg_confusion > 0.5 ? '#f85149' : '#d29922',
                                borderRadius: '4px'
                              }}></div>
                            </div>
                            <span style={{ fontSize: '14px' }}>{c.avg_confusion.toFixed(3)}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p style={{ color: '#8b949e' }}>No high-risk students found for this course.</p>
              )}
            </div>
          </div>

          {/* Simulation Results — Task D */}
          {report.simulation && (
            <div className="card">
              <h2>Intervention Simulation</h2>
              <p style={{ color: '#c9d1d9', fontSize: '14px', marginBottom: '6px' }}>
                "What if we add tutoring for this concept?" — simulated by modifying student features and re-running the GNN (Task D)
              </p>
              <p style={{ color: '#c9d1d9', fontSize: '15px', marginBottom: '20px' }}>
                Baseline high-risk students: <strong style={{ color: '#f85149', fontSize: '18px' }}>{report.simulation.baseline_high_risk}</strong>
              </p>

              {report.simulation.interventions && report.simulation.interventions.length > 0 ? (
                <div>
                  {report.simulation.interventions.map((item, i) => {
                    const maxHelped = Math.max(...report.simulation.interventions.map(x => x.students_helped));
                    const barWidth = maxHelped > 0 ? (item.students_helped / maxHelped) * 100 : 0;

                    return (
                      <div key={i} style={{
                        padding: '16px 20px',
                        marginBottom: '12px',
                        borderRadius: '8px',
                        background: '#0d1117',
                        border: '1px solid #58a6ff33'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                          <div>
                            <span style={{ fontWeight: 700, color: '#e1e4e8', fontSize: '16px' }}>
                              {i + 1}. Add tutoring for {item.concept_name}
                            </span>
                            <span style={{ color: '#8b949e', fontSize: '13px', marginLeft: '8px' }}>
                              {item.concept_id}
                            </span>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            <span style={{ color: '#3fb950', fontWeight: 700, fontSize: '20px' }}>
                              {item.students_helped}
                            </span>
                            <span style={{ color: '#c9d1d9', fontSize: '14px', marginLeft: '6px' }}>
                              students helped
                            </span>
                          </div>
                        </div>

                        <div style={{ height: '8px', background: '#21262d', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${barWidth}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, #3fb950, #58a6ff)',
                            borderRadius: '4px',
                            transition: 'width 0.5s'
                          }}></div>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px' }}>
                          <span style={{ color: '#8b949e', fontSize: '13px' }}>
                            {item.students_affected} students affected
                          </span>
                          <span style={{ color: '#8b949e', fontSize: '13px' }}>
                            {report.simulation.baseline_high_risk - item.students_helped} would remain high-risk
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p style={{ color: '#8b949e' }}>No simulation data available.</p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default ProfessorDashboard;