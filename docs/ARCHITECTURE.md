# Architecture
```mermaid
flowchart TD
U[User]-->UI[Flask dashboard]
UI-->V[CSV validation]
V-->F[Numeric feature columns]
F-->M[Isolation Forest baseline]
M-->R[Prediction and anomaly score]
R-->T[Results table / CSV download]
G[GitHub]-->A[GitHub Actions: pytest + Docker build]
A-->D[Docker image]-->DH[Docker Hub release]-->K[Kubernetes Deployment]-->S[Service]
```
Three tiers: presentation (HTML/CSS rendered by Flask); application (validation, prediction, export); model/data (scikit-learn pipeline). This starter does not parse raw HDFS logs.
## Scope warning
The previously discussed Linear SVM + LSTM + VQC model set is not implemented in this runnable baseline. Add and evaluate it against the actual SRS after verifying the dataset schema and prediction unit. Disclose this deviation if the SRS requires those models.
