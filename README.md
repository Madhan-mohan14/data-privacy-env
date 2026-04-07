# DataPrivacyEnv

A reinforcement learning environment for training AI agents to detect and redact Personally Identifiable Information (PII) from real-world document formats. Built for the **Meta x HuggingFace OpenEnv Hackathon**.

---

## The Problem

Organizations handling user data face strict regulatory requirements under **GDPR**, **HIPAA**, and **CCPA**. Manual PII redaction is error-prone and does not scale. A single missed email address, SSN, or medical record in a data breach can result in heavy fines and loss of user trust.

DataPrivacyEnv trains agents to autonomously locate and redact PII across diverse document types — plain text logs, CSV databases, and structured JSON records — before sensitive data is exposed.

---

## Architecture

```
inference.py (Agent)
      │
      │  HTTP (reset / step)
      ▼
FastAPI Server  ──►  DataPrivacyEnvironment
  (server/app.py)       (task logic + reward)
      │
      ▼
Docker Container (HF Space)
```

- **Framework**: [OpenEnv](https://github.com/meta-pytorch/openenv) — standardized RL environment protocol
- **Server**: FastAPI, served via `uvicorn` on port 7860
- **Deployment**: Docker container on Hugging Face Spaces
- **Agent**: OpenAI-compatible client (works with any LLM via `API_BASE_URL`)
- **Package manager**: `uv` for fast, reproducible installs

---

## Tasks

| Task | Difficulty | File | PII Types |
|------|-----------|------|-----------|
| `task1_plaintext` | Easy | `server_logs.txt` | Email addresses, phone numbers |
| `task2_csv` | Medium | `customers.csv` | Full names, Social Security Numbers |
| `task3_json` | Hard | `medical_database.json` + `breach_report.txt` | Name, diagnosis, medication, SSN, phone, email, insurance ID, date of birth |

### Task 1 — Plain Text Logs
The agent reads a server log file and redacts all email addresses and phone numbers embedded in log lines.

### Task 2 — CSV Customer Database
The agent reads a CSV with 4 customer rows and must redact exactly 8 fields: 4 full names and 4 SSNs (format `XXX-XX-XXXX`). Emails and customer IDs must NOT be redacted.

### Task 3 — Medical JSON + Breach Report
The agent first reads `breach_report.txt` to identify the compromised Patient ID, then locates that patient's record in `medical_database.json` and redacts all 8 sensitive fields. Requires cross-file reasoning.

---

## Action Space

The agent has access to 4 tools, expressed as JSON:

### `read_file`
Read the contents of a file in the environment.
```json
{"tool": "read_file", "file_path": "server_logs.txt"}
```

### `redact_text`
Replace an exact string in a file with `[REDACTED]`.
```json
{
  "tool": "redact_text",
  "file_path": "server_logs.txt",
  "target_string": "john.doe@company.com",
  "replacement": "[REDACTED]"
}
```

### `submit`
Signal task completion and trigger final scoring.
```json
{"tool": "submit"}
```

---

## Reward Function

| Event | Reward |
|-------|--------|
| Correct redaction (first time) | `+0.20` |
| Duplicate redaction (already redacted) | `-0.05` |
| Wrong redaction (non-PII field) | `-0.20` |
| Early submit (missing required redactions) | `-1.00` to `-2.80` |
| `read_file` (neutral) | `0.00` |
| Successful `submit` after all redactions | `0.00` |

Max achievable reward per task: **1.60**

---

## Achieved Scores

| Task | Score | Status |
|------|-------|--------|
| `task1_plaintext` | 1.00 | Passed |
| `task2_csv` | 1.00 | Passed |
| `task3_json` | 0.72 | Passed |
| **Average** | **0.91** | **All tasks passed** |

Success threshold: 0.60 per task.

---

## Try the Playground

The environment is live on Hugging Face Spaces. You can interact with it directly via HTTP:

**Space URL**: https://huggingface.co/spaces/Maddy140605/dataprivacy-env

### Reset a task
```bash
curl -X POST https://maddy140605-dataprivacy-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_plaintext"}'
```

### Take a step
```bash
curl -X POST https://maddy140605-dataprivacy-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"message": "{\"tool\": \"read_file\", \"file_path\": \"server_logs.txt\"}"}'
```

### Get current state
```bash
curl https://maddy140605-dataprivacy-env.hf.space/state
```

---

## Run Locally

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker

### Setup
```bash
git clone https://github.com/Madhan-mohan14/data-privacy-env
cd data-privacy-env/data_privacy_env

# Install dependencies
uv sync

# Start the server
uvicorn server.app:app --reload
```

### Build Docker image
```bash
docker build -t dataprivacy-env:latest .
```

### Run tests
```bash
pytest
pytest --cov=. --cov-report=term-missing
```

---

## Run Inference

The inference script runs the PII redaction agent against all 3 tasks.

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | No | LLM API endpoint (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | No | Model identifier (default: `meta-llama/Llama-3.1-8B-Instruct`) |
| `HF_TOKEN` | **Yes** | Your HuggingFace API token |
| `IMAGE_NAME` | No | Docker image name (default: `dataprivacy-env:latest`) |

### Setup
```bash
cd data_privacy_env
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### Run
```bash
python inference.py
```

Expected output format:
```
[START] task=task1_plaintext env=dataprivacy-env model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"tool": "read_file", ...} reward=0.00 done=false error=null
[END] success=true steps=10 score=1.00 rewards=0.00,0.20,0.20,...
```

---

## Links

- **HuggingFace Space**: https://huggingface.co/spaces/Maddy140605/dataprivacy-env
- **GitHub Repository**: https://github.com/Madhan-mohan14/data-privacy-env
- **OpenEnv Framework**: https://github.com/meta-pytorch/openenv
