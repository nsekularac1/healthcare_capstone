# Integrated Intelligent Health Monitoring and Clinical Decision Support System

Capstone prototype integrating:

- statistical analysis and feature reasoning;
- machine learning for diabetes readmission prediction;
- deep learning for PAMAP2 activity recognition;
- generative explanations;
- bounded agentic orchestration.

## Important design boundary

The Diabetes 130-US Hospitals and PAMAP2 datasets represent different populations and are not joined at the patient level. Their outputs are integrated only at the system/application layer.

## Repository layout

- `data/` — raw and processed public datasets
- `notebooks/` — EDA, modeling experiments, and evaluation
- `src/` — reusable implementation code
- `models/` — trained model artifacts
- `tests/` — automated tests
- `diagrams/` — system architecture visuals
- `reports/` — final synthesis paper and related reporting assets
- `config.yaml` — project configuration

## Reproducibility

Create and activate the project environment, then generate the final dependency file from that working environment:

```bash
pip freeze > requirements.txt
```

Do not treat the starter `requirements.txt` as final until implementation is complete.
