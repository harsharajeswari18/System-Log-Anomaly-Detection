# Product Backlog
Story points: Fibonacci scale. Proposed owners: A=UI, B=ML/data/docs, C=tests/DevOps. Reconcile with submitted SRS/SAD/STP.
| ID | Story | Acceptance criteria | Priority | SP | Owner |
|---|---|---|---|---:|---|
| US-01 | User opens dashboard | Home returns 200 and explains CSV contract | P0 | 2 | A |
| US-02 | User uploads CSV | Valid input accepted; invalid input handled | P0 | 3 | A |
| US-03 | User sees anomaly predictions | Output has Normal/Anomaly and score | P0 | 5 | B |
| US-04 | User downloads predictions | CSV contains original data plus output fields | P1 | 2 | A |
| US-05 | Developer runs tests | pytest passes locally and on PR | P0 | 3 | C |
| US-06 | App runs in Docker | compose app works on localhost:5000 | P0 | 3 | C |
| US-07 | Team reviews code | Feature branch, PR, CI and peer review | P0 | 2 | All |
| US-08 | Team publishes image | v* tag publishes to Docker Hub via secrets | P1 | 3 | C |
| US-09 | Operator deploys Kubernetes | 2 replicas, probes and Service work | P1 | 5 | C |
| US-10 | Reviewer sees docs | Architecture, STP, demo, deviations committed | P0 | 3 | B |
| US-11 | Researcher processes real HDFS data | Actual schema and feature conversion verified | P0 | 8 | B |
| US-12 | Researcher compares SVM/LSTM/VQC | Models trained/evaluated with reproducible setup | P1 | 8 | B |
| US-13 | Reviewer sees model metrics | Held-out precision, recall, F1, confusion matrix | P1 | 5 | B |
