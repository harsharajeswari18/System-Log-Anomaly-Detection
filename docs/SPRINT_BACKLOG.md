# Two-Sprint Backlog
Assignment brief dates: Sprint 1 19–23 Oct 2026 (1-minute demo); Sprint 2 26–30 Oct 2026 (2-minute demo). Follow any updated instructor dates.
## Team
A: UI/upload/download. B: dataset/model/evaluation/docs. C: tests/CI/Docker/Kubernetes. Everyone participates in PR reviews.
## Sprint 1 goal
Working local dashboard and baseline predictions, tests and CI.
| Item | SP | Owner | Done when |
|---|---:|---|---|
| US-01 dashboard | 2 | A | Home loads |
| US-02 CSV upload | 3 | A | Valid and invalid cases handled |
| US-03 baseline prediction | 5 | B | Prediction and score shown |
| US-05 tests/CI | 3 | C | pytest and PR workflow green |
| US-07 PR workflow | 2 | All | Issues, branch and reviewed PRs used |
| US-10 docs draft | 2 | B | README/architecture draft committed |
**Sprint 1 total: 17 SP.** Demo: run demo, upload sample CSV, show predictions and CI.
## Sprint 2 goal
Export results, containerize, publish image, deploy Kubernetes, finish docs.
| Item | SP | Owner | Done when |
|---|---:|---|---|
| US-04 download CSV | 2 | A | Download includes predictions |
| US-06 Docker | 3 | C | Container runs at localhost:5000 |
| US-08 Docker Hub | 3 | C | Image tag published |
| US-09 Kubernetes | 5 | C | Deployment/Service healthy |
| US-10 docs/demo | 3 | B | STP, architecture, deviations and final demo ready |
**Core Sprint 2 total: 16 SP.** US-11/12 are research-critical; include as sprint commitments only if dataset/schema and time permit. Never mark done until implemented.
## Issue workflow
Create GitHub Issues from backlog, add to Project, set story points/owner/iteration; branch `feature/US-01-dashboard`; commit; push; open PR; wait for CI; get peer review; merge; then move issue to Done.
