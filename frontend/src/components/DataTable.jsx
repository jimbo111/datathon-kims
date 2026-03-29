/**
 * Simple data table for previewing dataset rows.
 */
export default function DataTable({ data, maxRows = 20 }) {
  if (!data || data.length === 0) return <p>No data loaded.</p>;

  const columns = Object.keys(data[0]);
  const rows = data.slice(0, maxRows);

  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((col) => (
                <td key={col}>{row[col] != null ? String(row[col]) : "—"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
