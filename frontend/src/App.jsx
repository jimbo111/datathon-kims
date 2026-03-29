import { useState } from "react";
import Dashboard from "./components/Dashboard";
import HealthDashboard from "./components/HealthDashboard";
import "./App.css";

function App() {
  const [view, setView] = useState("health");

  return (
    <>
      <nav className="app-nav">
        <button
          className={`nav-btn ${view === "health" ? "active" : ""}`}
          onClick={() => setView("health")}
        >
          Health Analysis
        </button>
        <button
          className={`nav-btn ${view === "data" ? "active" : ""}`}
          onClick={() => setView("data")}
        >
          Data Explorer
        </button>
      </nav>
      {view === "health" ? <HealthDashboard /> : <Dashboard />}
    </>
  );
}

export default App;
