???+ info
        Please ensure formatting, linting, and security checks pass before submitting a pull request.

## Code Formatting

The codebase is formatted using [black](https://github.com/psf/black)

To format the codebase, run the following command:

```bash
black .
```

Please ensure that the code is formatted before submitting a pull request.

## Linting

The codebase is linted using [flake8](https://flake8.pycqa.org/en/latest/)

To view the linting errors, run the following command:

```bash
flake8 .
```

## Security and Vulnerabilities

The codebase is scanned for security vulnerabilities using [bandit](https://github.com/PyCQA/bandit)

To scan the codebase for security vulnerabilities, run the following command:

```bash
bandit -r .
```