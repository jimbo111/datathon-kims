import { useState, useEffect } from "react";
import { getHealthSummary } from "../api/client";

export default function HealthSummaryTable() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealthSummary()
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="chart-error">Error: {error}</div>;
  if (!data) return <div className="chart-loading">Loading summary...</div>;

  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>State</th>
            <th>Food Access</th>
            <th>% Food Desert</th>
            <th>Diabetes %</th>
            <th>Obesity %</th>
            <th>Median Income</th>
            <th>Life Expectancy</th>
            <th>PCPs/100k</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.state}>
              <td>{row.state_name}</td>
              <td className={row.food_access_score < 55 ? "cell-bad" : ""}>{row.food_access_score}</td>
              <td className={row.pct_food_desert > 25 ? "cell-bad" : ""}>{row.pct_food_desert}%</td>
              <td className={row.diabetes_rate > 12 ? "cell-bad" : ""}>{row.diabetes_rate}%</td>
              <td className={row.obesity_rate > 35 ? "cell-bad" : ""}>{row.obesity_rate}%</td>
              <td>${row.median_income.toLocaleString()}</td>
              <td className={row.life_expectancy < 76 ? "cell-bad" : ""}>{row.life_expectancy}</td>
              <td>{row.pcp_per_100k ?? "N/A"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
