# Deviations and risks
1. Runnable baseline uses Isolation Forest, not Linear SVM + LSTM + VQC. Disclose this; do not claim those models are implemented.
2. Upload expects numeric features, not raw HDFS log lines.
3. Verify actual HDFS dataset schema, label mapping, prediction unit and leakage-safe split before training.
4. Anomaly score is not a calibrated probability.
5. No held-out evaluation is reported by the baseline; synthetic demo data is for demonstration only.
6. Replace Docker Hub username/tag in Kubernetes manifest and configure repository secrets.
