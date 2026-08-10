# Automated Testing

The repository includes focused unit and integration tests.

## Run all tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Run only local tests

These do not require OpenAI API access:

```powershell
.\.venv\Scripts\python.exe -m pytest -v -m "not integration"
```

## Run integration tests

These require saved model artifacts and an `OPENAI_API_KEY` in the project `.env` file:

```powershell
.\.venv\Scripts\python.exe -m pytest -v -m integration
```

## Test coverage

The suite checks:

- Diabetes preprocessing
- PAMAP2 preprocessing
- patient/subject leakage protection
- saved clinical-model inference
- LSTM architecture and saved-model inference
- deterministic medical-scope safeguards
- OpenAI API connectivity
- clinical tool routing
- wearable tool routing
- integrated two-tool routing
- safety fallback behavior
