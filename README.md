# System Log Anomaly Detection — Two-Sprint Starter
Flask dashboard for numeric feature CSV upload, anomaly flags, results preview and CSV download. Includes tests, GitHub Actions, Docker, Docker Hub publishing workflow and Kubernetes manifests.
**Important:** runnable baseline is Isolation Forest. It does not yet implement Linear SVM + LSTM + VQC or parse raw HDFS logs. Reconcile with submitted SRS/SAD/STP; document or implement the required work before claiming completion.
## Local run (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python app.py
```
Open http://127.0.0.1:5000 and click Run demo dataset. If activation is blocked, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` then `.venv\Scripts\python.exe app.py`.
## Docker
Start Docker Desktop, then `docker compose up --build -d`; open http://localhost:5000; health is `/health`; logs `docker compose logs -f`; stop `docker compose down`.
## Existing GitHub repository
Back up and check `git status`. Copy/merge package files into repo root without overwriting your submitted documents. Then:
```bash
git checkout -b feature/devops-sprint-package
git add .
git commit -m "Add anomaly detection MVP and DevOps pipeline"
git push -u origin feature/devops-sprint-package
```
Open PR, check Actions, request peer review and merge. Create issues from `docs/PRODUCT_BACKLOG.md` and add them to GitHub Project with sprint, story points and owners.
## Docker Hub
Create Docker Hub repo `system-log-anomaly-detection`. Manual:
```powershell
docker login
docker build -t YOUR_DOCKERHUB_USERNAME/system-log-anomaly-detection:v1.0.0 .
docker push YOUR_DOCKERHUB_USERNAME/system-log-anomaly-detection:v1.0.0
```
For automation, GitHub repository Settings → Secrets and variables → Actions: add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (access token, not password). Check workflow image name, merge, then `git tag v1.0.0` and `git push origin v1.0.0`.
## Kubernetes on Docker Desktop
Enable Kubernetes in Docker Desktop settings and wait for it to start. Edit `k8s/deployment.yaml` image username/tag. Run:
```powershell
kubectl apply -f k8s/deployment.yaml
kubectl get deployments
kubectl get pods
kubectl get services
kubectl rollout status deployment/anomaly-detection
kubectl port-forward service/anomaly-detection-service 5000:5000
```
Open http://localhost:5000 and `/health`; stop port-forward with Ctrl+C. Cleanup: `kubectl delete -f k8s/deployment.yaml`.
## Team of 3
A UI/upload/export; B dataset, models, metrics and docs; C tests/CI/container/Kubernetes. All members use issues, branches, PR review and evidence.
## Evidence to submit
GitHub Project/backlog, story points and assignees; two sprint boards; PRs and green Actions; Docker build/run; Docker Hub image/tag; Kubernetes pods/service/health; Sprint 1 one-minute and Sprint 2 two-minute videos uploaded to repo; updated SRS/SAD/STP, test results and deviations.
