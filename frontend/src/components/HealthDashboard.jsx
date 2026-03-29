import { useState, useEffect, useCallback } from "react";
import Chart from "./Chart";
import {
  getChoroplethFoodAccess,
  getChoroplethDiabetes,
  getChoroplethObesity,
  getChoroplethLifeExpectancy,
  getScatterFoodVsDiabetes,
  getScatterFoodVsObesity,
  getBarLifeExpectancyIncome,
  getHeatmapIncomeRaceDiabetes,
  getChoroplethHDI,
  getPathDiagram,
  getHdiRankedTracts,
} from "../api/client";
import HealthSummaryTable from "./HealthSummaryTable";

const TABS = [
  { id: "maps", label: "Choropleth Maps" },
  { id: "scatter", label: "Correlations" },
  { id: "race", label: "Race Gap" },
  { id: "synthesis", label: "Synthesis" },
];

export default function HealthDashboard() {
  const [activeTab, setActiveTab] = useState("maps");

  return (
    <div className="dashboard">
      <header>
        <h1>Your Zip Code is Your Health Sentence</h1>
        <p className="subtitle">
          Exploring how geography, food access, and income shape health outcomes
          across the United States
        </p>
      </header>

      <nav className="tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === "maps" && <MapsTab />}
      {activeTab === "scatter" && <ScatterTab />}
      {activeTab === "race" && <RaceGapTab />}
      {activeTab === "synthesis" && <SynthesisTab />}
    </div>
  );
}

function MapsTab() {
  const fetchFoodAccess = useCallback(() => getChoroplethFoodAccess(), []);
  const fetchDiabetes = useCallback(() => getChoroplethDiabetes(), []);
  const fetchObesity = useCallback(() => getChoroplethObesity(), []);
  const fetchLifeExp = useCallback(() => getChoroplethLifeExpectancy(), []);

  return (
    <>
      <section className="panel">
        <h2>Side-by-Side: Food Access vs Disease</h2>
        <p className="panel-desc">
          States with lower food access scores (red) cluster with higher diabetes
          rates — food deserts and chronic disease are not randomly distributed.
        </p>
        <div className="chart-grid two-col">
          <Chart fetchFn={fetchFoodAccess} title="Food Access Score" style={{ height: 450 }} />
          <Chart fetchFn={fetchDiabetes} title="Diabetes Rate" style={{ height: 450 }} />
        </div>
      </section>

      <section className="panel">
        <h2>Obesity and Life Expectancy</h2>
        <p className="panel-desc">
          The obesity map mirrors diabetes patterns. Life expectancy reveals the
          ultimate cost — up to 10 years of life lost between the best and worst states.
        </p>
        <div className="chart-grid two-col">
          <Chart fetchFn={fetchObesity} title="Obesity Rate" style={{ height: 450 }} />
          <Chart fetchFn={fetchLifeExp} title="Life Expectancy" style={{ height: 450 }} />
        </div>
      </section>
    </>
  );
}

function ScatterTab() {
  const fetchFoodDiabetes = useCallback(() => getScatterFoodVsDiabetes(), []);
  const fetchFoodObesity = useCallback(() => getScatterFoodVsObesity(), []);

  return (
    <section className="panel">
      <h2>Food Access vs Chronic Disease</h2>
      <p className="panel-desc">
        Each dot is a state. Size = median income, color = % minority population.
        The trend line confirms: lower food access scores correlate with higher
        disease rates, even when accounting for income.
      </p>
      <div className="chart-grid two-col">
        <Chart fetchFn={fetchFoodDiabetes} title="Food Access vs Diabetes" style={{ height: 450 }} />
        <Chart fetchFn={fetchFoodObesity} title="Food Access vs Obesity" style={{ height: 450 }} />
      </div>
    </section>
  );
}

function RaceGapTab() {
  const fetchHeatmap = useCallback(() => getHeatmapIncomeRaceDiabetes(), []);
  const fetchLifeExpIncome = useCallback(() => getBarLifeExpectancyIncome(), []);

  return (
    <>
      <section className="panel">
        <h2>Income x Race → Diabetes Prevalence</h2>
        <p className="panel-desc">
          This heatmap reveals that race compounds the income effect on diabetes.
          Compare the richest majority-Black tracts to the poorest majority-White
          tracts — income alone does not close the racial health gap.
        </p>
        <div className="chart-grid">
          <Chart fetchFn={fetchHeatmap} title="Income × Race Heatmap" style={{ height: 450 }} />
        </div>
      </section>

      <section className="panel">
        <h2>Life Expectancy by Income Quintile</h2>
        <p className="panel-desc">
          States in the lowest income quintile live an average of 4+ fewer years
          than the highest. The gap is a direct measure of structural inequality.
        </p>
        <div className="chart-grid">
          <Chart fetchFn={fetchLifeExpIncome} title="Life Expectancy by Income" style={{ height: 400 }} />
        </div>
      </section>
    </>
  );
}

function SynthesisTab() {
  const fetchHDI = useCallback(() => getChoroplethHDI(), []);
  const fetchPathDiagram = useCallback(() => getPathDiagram(), []);
  const [worstTracts, setWorstTracts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHdiRankedTracts(10)
      .then(setWorstTracts)
      .catch(() => setWorstTracts([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <>
      <section className="panel">
        <h2>Health Disadvantage Index</h2>
        <p className="panel-desc">
          A composite index combining food access, poverty rate, and healthcare
          access (equal-weighted). Red states face the greatest compounding
          disadvantage across all three dimensions.
        </p>
        <div className="chart-grid">
          <Chart fetchFn={fetchHDI} title="Health Disadvantage Index" style={{ height: 450 }} />
        </div>
      </section>

      <section className="panel">
        <h2>Pathway: Poverty → Food Desert → Obesity → Diabetes</h2>
        <p className="panel-desc">
          Bivariate associations along the hypothesized causal pathway. Each arrow
          shows the regression coefficient (β) and R². Note: these are independent
          associations, not a formal mediation analysis.
        </p>
        <div className="chart-grid">
          <Chart fetchFn={fetchPathDiagram} title="Path Diagram" style={{ height: 380 }} />
        </div>
      </section>

      <section className="panel">
        <h2>Most Disadvantaged Census Tracts</h2>
        <p className="panel-desc">
          Top 10 tracts with the highest Health Disadvantage Index score — these
          are the places where zip code most constrains health outcomes.
        </p>
        {loading ? (
          <p style={{ color: "#8b949e" }}>Loading...</p>
        ) : worstTracts.length > 0 ? (
          <table className="summary-table">
            <thead>
              <tr>
                <th>Tract FIPS</th>
                <th>State</th>
                <th>HDI Score</th>
                <th>Diabetes %</th>
                <th>Life Expectancy</th>
              </tr>
            </thead>
            <tbody>
              {worstTracts.map((t, i) => (
                <tr key={i}>
                  <td>{t.tract_fips}</td>
                  <td>{t.state || "—"}</td>
                  <td>{t.hdi_score?.toFixed(2)}</td>
                  <td className={t.diabetes_pct > 15 ? "cell-bad" : ""}>{t.diabetes_pct?.toFixed(1)}</td>
                  <td className={t.life_expectancy < 72 ? "cell-bad" : ""}>{t.life_expectancy?.toFixed(1) || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ color: "#8b949e" }}>No HDI data available. Run the analysis pipeline first.</p>
        )}
      </section>

      <section className="panel">
        <h2>Most Disadvantaged States</h2>
        <p className="panel-desc">
          Bottom 10 states by food access score — aggregated from census tract data.
        </p>
        <HealthSummaryTable />
      </section>
    </>
  );
}
