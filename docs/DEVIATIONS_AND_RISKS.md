# Deviations and risks
1. Linear SVM (scikit-learn), LSTM (pure NumPy, manual BPTT, 4 features fed as a length-4 sequence) and VQC (4-qubit circuit simulated in NumPy, parameter-shift training) are implemented and compared against an Isolation Forest baseline. The VQC is a classical simulation, not run on quantum hardware; the LSTM is a small model on tabular features, not on raw log sequences.
2. Upload expects numeric features, not raw HDFS log lines.
3. Verify actual HDFS dataset schema, label mapping, prediction unit and leakage-safe split before training.
4. Anomaly score is not a calibrated probability.
5. No held-out evaluation is reported by the baseline; synthetic demo data is for demonstration only.
6. Replace Docker Hub username/tag in Kubernetes manifest and configure repository secrets.
