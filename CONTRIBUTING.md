# Contribution & Branch Workflow

All future changes to `ai-dashboard` must follow a clean feature-branch + Pull Request workflow:

1. Create a new branch from `main` (e.g. `feat/<topic>` or `fix/<topic>`).
2. Run `go test -race ./...` and verify JS/UI checks pass before pushing.
3. Push the feature branch to `origin` and open a Pull Request into `main`.
4. Do not push directly to `main` once branch protection is active.
