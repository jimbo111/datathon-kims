import { useState, useEffect } from "react";
import PlotModule from "react-plotly.js";

// Handle CJS default export compatibility
const Plot = PlotModule.default || PlotModule;

/**
 * Reusable chart component — takes a fetch function that returns Plotly JSON.
 *
 * Usage:
 *   <Chart fetchFn={() => getDistribution("age")} title="Age Distribution" />
 */
export default function Chart({ fetchFn, title, style = {} }) {
  const [figure, setFigure] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchFn()
      .then((fig) => {
        if (!cancelled) setFigure(fig);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [fetchFn]);

  if (loading) return <div className="chart-loading">Loading {title}...</div>;
  if (error) return <div className="chart-error">Error: {error}</div>;
  if (!figure) return null;

  return (
    <div className="chart-container" style={style}>
      <Plot
        data={figure.data}
        layout={{
          ...figure.layout,
          autosize: true,
          margin: { t: 40, r: 20, b: 40, l: 60 },
        }}
        config={{ responsive: true, displayModeBar: true }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
