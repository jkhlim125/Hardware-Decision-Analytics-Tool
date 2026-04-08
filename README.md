📊 Experiment Log Anomaly Dashboard

An interactive analytics dashboard for monitoring experiment performance, detecting anomalies, and improving reproducibility.

⸻

** Overview

Machine learning experiments often produce inconsistent results across runs.
This project provides a dashboard to:
	•	Detect anomalous runs using statistical methods
	•	Compare performance across experiments
	•	Analyze correlations between metrics
	•	Generate automatic summaries for debugging

⸻

** Features

	•	📈 Run-level performance comparison
	•	🚨 Anomaly detection (Z-score based)
	•	🔍 Metric selection & filtering
	•	📊 Scatter plots & correlation heatmap
	•	📉 Normalized comparison (0–1 scaling / Z-score)
	•	🧠 Automatic summary generation
	•	📥 Export results (CSV)
  
____

** How to Run Locally

pip install -r requirements.txt
streamlit run app.py

____

** Project Motivation

Experiments are often hard to debug due to variability across runs.
This dashboard helps identify performance issues and improve reproducibility.

⸻

** Key Insight

	•	Automatically flags abnormal runs based on statistical deviation
	•	Enables fast comparison between runs and average performance
	•	Helps identify trade-offs between accuracy, latency, and memory
