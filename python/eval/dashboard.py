"""
HTML eval dashboard — visual retrieval quality report.

Generates a self-contained HTML file showing:
  - Faithfulness and relevance scores per query
  - Pass/fail distribution (FAITHFUL vs HALLUCINATED)
  - Latency and cost breakdown
  - Strategy comparison (vector vs hybrid vs reranked)

No external dependencies — pure HTML/CSS/JS with inline Chart.js from CDN.

Usage:
    from python.eval.dashboard import EvalDashboard
    dashboard = EvalDashboard()
    dashboard.add_results(eval_results)
    dashboard.save("rag_eval_report.html")
    # Open rag_eval_report.html in any browser
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .metrics import EvalResult


@dataclass
class DashboardConfig:
    title: str = "RAG Evaluation Report"
    theme: str = "light"


class EvalDashboard:
    """
    Generate a self-contained HTML eval report.

    Usage:
        dashboard = EvalDashboard()
        for query, answer, retrieval in zip(queries, answers, retrievals):
            result = evaluator.evaluate(query, answer, retrieval)
            dashboard.add(result)
        dashboard.save("report.html")
    """

    def __init__(self, config: Optional[DashboardConfig] = None) -> None:
        self.config = config or DashboardConfig()
        self._results: list[EvalResult] = []

    def add(self, result: EvalResult) -> None:
        self._results.append(result)

    def add_results(self, results: list[EvalResult]) -> None:
        self._results.extend(results)

    def save(self, path: str = "rag_eval_report.html") -> str:
        html = self._render()
        Path(path).write_text(html, encoding="utf-8")
        return path

    def _summary_stats(self) -> dict:
        if not self._results:
            return {}
        n = len(self._results)
        return {
            "total": n,
            "avg_faithfulness": round(sum(r.faithfulness.score for r in self._results) / n, 3),
            "avg_relevance": round(sum(r.relevance.score for r in self._results) / n, 3),
            "avg_overall": round(sum(r.overall_score for r in self._results) / n, 3),
            "faithful_count": sum(1 for r in self._results if r.faithfulness.verdict == "FAITHFUL"),
            "hallucinated_count": sum(1 for r in self._results if r.faithfulness.verdict == "HALLUCINATED"),
            "relevant_count": sum(1 for r in self._results if r.relevance.verdict == "RELEVANT"),
            "irrelevant_count": sum(1 for r in self._results if r.relevance.verdict == "IRRELEVANT"),
            "avg_latency_ms": round(sum(r.retrieval_latency_ms for r in self._results) / n, 1),
            "total_cost_usd": round(sum(r.cost_usd for r in self._results), 4),
        }

    def _render(self) -> str:
        stats = self._summary_stats()
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows_data = json.dumps([
            {
                "query": r.query[:80],
                "faithfulness": round(r.faithfulness.score, 2),
                "relevance": round(r.relevance.score, 2),
                "overall": round(r.overall_score, 2),
                "f_verdict": r.faithfulness.verdict,
                "r_verdict": r.relevance.verdict,
                "latency_ms": round(r.retrieval_latency_ms, 1),
                "cost": round(r.cost_usd, 4),
            }
            for r in self._results
        ])

        faith_dist = json.dumps({
            "FAITHFUL": stats.get("faithful_count", 0),
            "PARTIAL": stats.get("total", 0) - stats.get("faithful_count", 0) - stats.get("hallucinated_count", 0),
            "HALLUCINATED": stats.get("hallucinated_count", 0),
        })

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self.config.title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f8f9fa; color: #1a1a1a; padding: 2rem; }}
  .header {{ background: linear-gradient(135deg, #4285F4, #34A853);
             color: white; border-radius: 12px; padding: 2rem; margin-bottom: 2rem; }}
  .header h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; }}
  .header p {{ opacity: 0.85; font-size: 0.9rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem; margin-bottom: 2rem; }}
  .stat {{ background: white; border-radius: 10px; padding: 1.2rem;
           box-shadow: 0 1px 4px rgba(0,0,0,0.07); text-align: center; }}
  .stat .value {{ font-size: 2rem; font-weight: 700; color: #4285F4; }}
  .stat .label {{ font-size: 0.8rem; color: #888; margin-top: 0.3rem; text-transform: uppercase; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-card {{ background: white; border-radius: 10px; padding: 1.5rem;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.07); }}
  .chart-card h3 {{ font-size: 0.9rem; text-transform: uppercase; color: #888;
                    margin-bottom: 1rem; letter-spacing: 0.05em; }}
  .table-card {{ background: white; border-radius: 10px; padding: 1.5rem;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.07); overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ background: #4285F4; color: white; padding: 0.6rem 0.8rem; text-align: left; }}
  td {{ padding: 0.55rem 0.8rem; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #fafbff; }}
  .badge {{ border-radius: 20px; padding: 0.2rem 0.6rem; font-size: 0.75rem; font-weight: 600; }}
  .FAITHFUL {{ background: #e6f4ea; color: #1e7e34; }}
  .PARTIAL {{ background: #fff8e1; color: #856404; }}
  .HALLUCINATED {{ background: #fce8e6; color: #c0392b; }}
  .RELEVANT {{ background: #e6f4ea; color: #1e7e34; }}
  .IRRELEVANT {{ background: #fce8e6; color: #c0392b; }}
  footer {{ text-align: center; color: #aaa; font-size: 0.8rem; margin-top: 2rem; }}
  @media (max-width: 700px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="header">
  <h1>{self.config.title}</h1>
  <p>Generated: {generated_at} &nbsp;|&nbsp; {stats.get("total", 0)} queries evaluated</p>
</div>

<div class="stats">
  <div class="stat"><div class="value">{stats.get("avg_overall", 0):.2f}</div><div class="label">Overall Score</div></div>
  <div class="stat"><div class="value">{stats.get("avg_faithfulness", 0):.2f}</div><div class="label">Faithfulness</div></div>
  <div class="stat"><div class="value">{stats.get("avg_relevance", 0):.2f}</div><div class="label">Relevance</div></div>
  <div class="stat"><div class="value">{stats.get("avg_latency_ms", 0):.0f}ms</div><div class="label">Avg Latency</div></div>
  <div class="stat"><div class="value">${stats.get("total_cost_usd", 0):.4f}</div><div class="label">Total Cost</div></div>
  <div class="stat"><div class="value">{stats.get("hallucinated_count", 0)}</div><div class="label">Hallucinated</div></div>
</div>

<div class="charts">
  <div class="chart-card">
    <h3>Score Distribution</h3>
    <canvas id="scoreChart" height="200"></canvas>
  </div>
  <div class="chart-card">
    <h3>Faithfulness Breakdown</h3>
    <canvas id="faithChart" height="200"></canvas>
  </div>
</div>

<div class="table-card">
  <h3 style="font-size:0.9rem;text-transform:uppercase;color:#888;letter-spacing:.05em;margin-bottom:1rem;">Query Results</h3>
  <table id="resultsTable">
    <thead>
      <tr>
        <th>Query</th>
        <th>Faithfulness</th>
        <th>Relevance</th>
        <th>Overall</th>
        <th>Latency</th>
        <th>Verdict</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
</div>

<footer>rag-patterns eval dashboard &nbsp;|&nbsp; github.com/TushGoel/rag-patterns</footer>

<script>
const rows = {rows_data};
const faithDist = {faith_dist};

const tbody = document.getElementById("tableBody");
rows.forEach(r => {{
  tbody.innerHTML += `<tr>
    <td title="${{r.query}}">${{r.query.length > 60 ? r.query.slice(0,60)+"..." : r.query}}</td>
    <td>${{r.faithfulness}}</td>
    <td>${{r.relevance}}</td>
    <td><strong>${{r.overall}}</strong></td>
    <td>${{r.latency_ms}}ms</td>
    <td><span class="badge ${{r.f_verdict}}">${{r.f_verdict}}</span></td>
  </tr>`;
}});

new Chart(document.getElementById("scoreChart"), {{
  type: "bar",
  data: {{
    labels: rows.map((_, i) => `Q${{i+1}}`),
    datasets: [
      {{ label: "Faithfulness", data: rows.map(r => r.faithfulness), backgroundColor: "#4285F4aa" }},
      {{ label: "Relevance", data: rows.map(r => r.relevance), backgroundColor: "#34A853aa" }},
      {{ label: "Overall", data: rows.map(r => r.overall), backgroundColor: "#EA4335aa" }},
    ]
  }},
  options: {{ scales: {{ y: {{ min: 0, max: 1 }} }}, plugins: {{ legend: {{ position: "bottom" }} }} }}
}});

new Chart(document.getElementById("faithChart"), {{
  type: "doughnut",
  data: {{
    labels: Object.keys(faithDist),
    datasets: [{{ data: Object.values(faithDist),
      backgroundColor: ["#34A853", "#FFC107", "#EA4335"] }}]
  }},
  options: {{ plugins: {{ legend: {{ position: "bottom" }} }} }}
}});
</script>
</body>
</html>"""
