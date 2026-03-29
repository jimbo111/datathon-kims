import { useState, useEffect, useCallback } from "react";
import Chart from "./Chart";
import DataTable from "./DataTable";
import {
  listDatasets,
  loadDataset,
  getDataInfo,
  getDataHead,
  getColumns,
  getDistribution,
  getCorrelation,
  getBarChart,
} from "../api/client";

export default function Dashboard() {
  const [datasets, setDatasets] = useState({});
  const [loaded, setLoaded] = useState(null);
  const [info, setInfo] = useState(null);
  const [preview, setPreview] = useState(null);
  const [cols, setCols] = useState({ numeric: [], categorical: [], datetime: [] });

  useEffect(() => {
    listDatasets().then(setDatasets).catch(console.error);
  }, []);

  const handleLoad = async (filename, subdir) => {
    const result = await loadDataset(filename, subdir);
    setLoaded({ name: filename, shape: result.shape });

    const [infoRes, headRes, colRes] = await Promise.all([
      getDataInfo(),
      getDataHead(20),
      getColumns(),
    ]);
    setInfo(infoRes);
    setPreview(headRes);
    setCols(colRes);
  };

  return (
    <div className="dashboard">
      <header>
        <h1>Datathon 2026 — Team 2Kim</h1>
      </header>

      {/* Dataset selector */}
      <section className="panel">
        <h2>Datasets</h2>
        {Object.keys(datasets).length === 0 ? (
          <p>No datasets found. Drop files into <code>data/raw/</code></p>
        ) : (
          Object.entries(datasets).map(([subdir, files]) => (
            <div key={subdir}>
              <h3>{subdir}/</h3>
              <div className="dataset-list">
                {files.map((f) => (
                  <button key={f} onClick={() => handleLoad(f, subdir)}>
                    {f}
                  </button>
                ))}
              </div>
            </div>
          ))
        )}
      </section>

      {loaded && (
        <>
          <section className="panel">
            <h2>
              {loaded.name}{" "}
              <span className="badge">
                {loaded.shape[0].toLocaleString()} rows × {loaded.shape[1]} cols
              </span>
            </h2>
            <DataTable data={preview} />
          </section>

          {/* Auto-generated charts */}
          <section className="panel">
            <h2>Overview</h2>
            <div className="chart-grid">
              {cols.numeric.length > 1 && (
                <Chart
                  fetchFn={useCallback(() => getCorrelation(), [])}
                  title="Correlation Matrix"
                  style={{ height: 500 }}
                />
              )}
              {cols.numeric.slice(0, 4).map((col) => (
                <Chart
                  key={`dist-${col}`}
                  fetchFn={useCallback(() => getDistribution(col), [col])}
                  title={`Distribution: ${col}`}
                  style={{ height: 350 }}
                />
              ))}
              {cols.categorical.slice(0, 4).map((col) => (
                <Chart
                  key={`bar-${col}`}
                  fetchFn={useCallback(() => getBarChart(col), [col])}
                  title={`Counts: ${col}`}
                  style={{ height: 350 }}
                />
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
