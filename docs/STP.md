# Software Test Plan
Scope: dashboard, CSV upload, baseline prediction, health endpoint, export, CI, Docker and Kubernetes. This does not prove research accuracy or raw HDFS parsing.
| ID | Test | Steps | Expected |
|---|---|---|---|
| TC-01 | Home | GET / | HTTP 200 |
| TC-02 | Demo | POST /demo | Summary and predictions |
| TC-03 | Upload | Upload numeric CSV | Result preview shown |
| TC-04 | Negative | Upload malformed/empty CSV | Friendly error; app stays up |
| TC-05 | Pipeline | Run prediction function | prediction and anomaly_score columns |
| TC-06 | Health | GET /health | JSON status ok |
| TC-07 | Export | Click download | CSV includes prediction fields |
| TC-08 | CI | Open PR | GitHub Actions passes |
| TC-09 | Docker | docker compose up --build | localhost:5000 responds |
| TC-10 | Kubernetes | Apply manifests, port-forward | /health returns 200 |
| TC-11 | HDFS | Process actual dataset | Pending verified schema and preprocessing |
| TC-12 | Metrics | Precision/recall/F1 | Pending agreed split and labels |
Exit criteria: MVP tests pass, CI green, image builds, deployment documented; research criteria remain open until implemented.
