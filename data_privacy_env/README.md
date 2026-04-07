---
title: DataPrivacyEnv
emoji: 🔒
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - privacy
  - pii
  - gdpr
---

# DataPrivacyEnv

PII Redaction and GDPR Compliance RL Environment
Built for Meta x HuggingFace OpenEnv Hackathon

## What it does
An RL environment where an AI agent learns to find and 
redact Personally Identifiable Information (PII) from 
files without breaking file structure. Simulates real 
GDPR/HIPAA compliance work.

## Action Space
Agent sends JSON strings with one of 4 tools:
- list_files: {"tool": "list_files", "directory": "."}
- read_file: {"tool": "read_file", "file_path": "file.txt"}
- redact_text: {"tool": "redact_text", "file_path": "f.txt", 
               "target_string": "john@email.com", 
               "replacement": "[REDACTED]"}
- submit: {"tool": "submit"}

## Observation Space
- task_id: current task identifier
- task_description: what the agent must do
- available_tools: list of tools
- last_action_result: result of last action
- last_reward: reward from last step
- cumulative_reward: total reward so far
- files_in_scope: files available in this task
- step_number: current step
- max_steps: maximum steps allowed
- done: whether episode is complete

## Reward Function
- +0.2 correct PII redaction (true positive)
- -0.2 false positive (redacted non-PII value)
- -0.05 invalid action (file not found, bad JSON)
- -0.3 per missed PII on submit
- -1.0 immediate fail if CSV/JSON structure broken

## Tasks
### Task 1 — Easy (task1_plaintext)
Redact 4 email addresses and 4 phone numbers from 
server_logs.txt. Max reward: 1.6

### Task 2 — Medium (task2_csv)
Redact name and SSN columns from customers.csv without 
breaking CSV structure. Emails and IDs must be preserved.
Max reward: 1.6

### Task 3 — Hard (task3_json)
Cross-file contextual redaction. Read breach_report.txt 
to find compromised Patient ID, then redact all sensitive 
fields (name, diagnosis, medication, ssn, phone, email, 
insurance_id, date_of_birth) for that patient in 
medical_database.json. Max reward: 1.6

## Setup
```bash
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Run Inference
```bash
cp .env.example .env
# Edit .env with your API key and model
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=your_groq_or_hf_token
export MODEL_NAME=llama-3.3-70b-versatile
export IMAGE_NAME=dataprivacy-env:latest
python inference.py
```

## Docker
```bash
docker build -t dataprivacy-env:latest .
docker run -p 8000:8000 dataprivacy-env:latest
```

## Achieved Scores (llama-3.3-70b-versatile via Groq)
- task1_plaintext: 1.00 (perfect)
- task2_csv: 1.00 (perfect)
- task3_json: 0.72 (hard cross-file task)
- **Average: 0.91**
