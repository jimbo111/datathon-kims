import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 30000,
});

// --- Data ---
export const listDatasets = () => api.get("/data/datasets").then((r) => r.data);
export const loadDataset = (filename, subdir = "raw") =>
  api.post(`/data/load/${filename}?subdir=${subdir}`).then((r) => r.data);
export const getDataInfo = () => api.get("/data/info").then((r) => r.data);
export const getDataHead = (n = 20) =>
  api.get("/data/head", { params: { n } }).then((r) => r.data);
export const getColumns = () => api.get("/data/columns").then((r) => r.data);
export const getStats = () => api.get("/data/stats").then((r) => r.data);

// --- Charts ---
export const getDistribution = (column, bins = 30) =>
  api.get("/charts/distribution", { params: { column, bins } }).then((r) => r.data);
export const getCorrelation = (method = "pearson") =>
  api.get("/charts/correlation", { params: { method } }).then((r) => r.data);
export const getBarChart = (column, topN = 20) =>
  api.get("/charts/bar", { params: { column, top_n: topN } }).then((r) => r.data);
export const getScatter = (x, y, hue = null) =>
  api.get("/charts/scatter", { params: { x, y, hue } }).then((r) => r.data);
export const getTimeSeries = (dateCol, valueCol, freq = null) =>
  api.get("/charts/timeseries", { params: { date_col: dateCol, value_col: valueCol, freq } }).then((r) => r.data);

// --- Health Analysis Maps ---
export const getChoroplethFoodAccess = () =>
  api.get("/charts/health/choropleth/food-access").then((r) => r.data);
export const getChoroplethDiabetes = () =>
  api.get("/charts/health/choropleth/diabetes").then((r) => r.data);
export const getChoroplethObesity = () =>
  api.get("/charts/health/choropleth/obesity").then((r) => r.data);
export const getChoroplethLifeExpectancy = () =>
  api.get("/charts/health/choropleth/life-expectancy").then((r) => r.data);
export const getScatterFoodVsDiabetes = () =>
  api.get("/charts/health/scatter/food-vs-diabetes").then((r) => r.data);
export const getScatterFoodVsObesity = () =>
  api.get("/charts/health/scatter/food-vs-obesity").then((r) => r.data);
export const getBarLifeExpectancyIncome = () =>
  api.get("/charts/health/bar/life-expectancy-income").then((r) => r.data);
export const getHealthSummary = () =>
  api.get("/charts/health/summary").then((r) => r.data);

// --- Phase 4 + 5A ---
export const getHeatmapIncomeRaceDiabetes = () =>
  api.get("/charts/health/heatmap/income-race-diabetes").then((r) => r.data);
export const getChoroplethHDI = () =>
  api.get("/charts/health/choropleth/hdi").then((r) => r.data);
export const getPathDiagram = () =>
  api.get("/charts/health/path-diagram").then((r) => r.data);
export const getHdiRankedTracts = (topN = 10) =>
  api.get("/data/hdi-ranked", { params: { top_n: topN } }).then((r) => r.data);

export default api;
