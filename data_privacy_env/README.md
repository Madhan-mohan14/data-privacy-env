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
insurance_id) for that patient in medical_database.json.
Max reward: 1.6

## Setup
```bash
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Run Inference
```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export IMAGE_NAME=dataprivacy-env:latest
python inference.py
```

## Docker
```bash
docker build -t dataprivacy-env:latest -f server/Dockerfile .
docker run -p 7860:7860 dataprivacy-env:latest
```

## Baseline Scores
- task1_plaintext: 0.62 (easy)
- task2_csv: 0.51 (medium)
- task3_json: 0.38 (hard)
