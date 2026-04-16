import { useState } from 'react';
import './App.css';
import StudentDashboard from './StudentDashboard';
import ProfessorDashboard from './ProfessorDashboard';

function App() {
  const [activeTab, setActiveTab] = useState('student');

  return (
    <div className="app">
      <header className="header">
        <h1>PathPilot</h1>
        <p>AI-Powered Student Learning Guidance</p>
        <nav className="tabs">
          <button
            className={activeTab === 'student' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('student')}
          >
            Student Dashboard
          </button>
          <button
            className={activeTab === 'professor' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('professor')}
          >
            Professor Dashboard
          </button>
        </nav>
      </header>

      <main className="content">
        {activeTab === 'student' ? <StudentDashboard /> : <ProfessorDashboard />}
      </main>
    </div>
  );
}

export default App;