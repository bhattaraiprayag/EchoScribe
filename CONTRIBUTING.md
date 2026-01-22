# Contributing to EchoScribe

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to EchoScribe. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by a Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for EchoScribe. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

- **Use a clear and descriptive title** for the issue to identify the problem.
- **Describe the exact steps which reproduce the problem** in as much detail as possible.
- **Provide specific examples** to demonstrate the steps.
- **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
- **Explain which behavior you expected to see instead and why.**

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for EchoScribe, including completely new features and minor improvements to existing functionality.

- **Use a clear and descriptive title** for the issue to identify the suggestion.
- **Provide a step-by-step description of the suggested enhancement** in as much detail as possible.
- **Provide specific examples to demonstrate the steps**.

### Pull Requests

The process described here has several goals:

- Maintain EchoScribe's quality.
- Fix problems that are important to users.
- Engage the community in working toward the best possible EchoScribe.

1.  **Fork the repo** and create your branch from `main`.
2.  **Run dependencies installation** using `uv sync`.
3.  **Run pre-commit hooks** locally to ensure your code is linted and formatted correctly.
    ```bash
    uv run pre-commit run --all-files
    ```
4.  **Test your changes** to ensure nothing is broken.
    ```bash
    uv run pytest
    ```
5.  **Run the application** locally to verify changes.
    ```bash
    uv run uvicorn backend.main:app --reload
    ```
6.  **Push** to your fork and submit a Pull Request.

## Development Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (Package Manager)
- Python 3.11+
- Docker (optional, but recommended for testing builds)

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/bhattaraiprayag/EchoScribe.git
    cd EchoScribe
    ```

2.  Install dependencies:

    ```bash
    uv sync
    ```

3.  Install pre-commit hooks:
    ```bash
    uv run pre-commit install
    ```

### Coding Standards

- **Python**: We use `ruff` for linting and formatting.
- **Frontend**: We use `prettier` for HTML/JS/CSS formatting.
- **Testing**: We use `pytest`. All new features must include tests.
- **Commits**: Write clear, descriptive commit messages.

## Infrastructure / DevOps

- **CI/CD**: We use GitHub Actions for CI. Ensure existing workflows pass.
- **Docker**: We use multi-stage builds. If you modify the `Dockerfile`, ensure optimization and security best practices (non-root user, etc.).

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
