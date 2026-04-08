# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import csv
import io
import random
import string
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataPrivacyAction, DataPrivacyObservation
except ImportError:
    from models import DataPrivacyAction, DataPrivacyObservation

# ---------------------------------------------------------------------------
# PII pools for randomization — each episode generates fresh data
# ---------------------------------------------------------------------------
_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Maria", "Nathan", "Olivia", "Peter",
    "Quinn", "Rachel", "Samuel", "Tina",
]
_LAST_NAMES = [
    "Johnson", "Martinez", "White", "Brown", "Davis", "Wilson",
    "Anderson", "Taylor", "Thomas", "Jackson", "Harris", "Clark",
    "Lewis", "Robinson", "Walker", "Hall",
]
_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "company.com", "email.org",
    "domain.co.uk", "internal.org", "private.com", "work.net",
    "corp.io", "secure.org",
]
_COLORS = ["blue", "green", "red", "purple", "orange", "yellow", "pink", "cyan"]
_DIAGNOSES = [
    "Type 2 Diabetes", "Hypertension", "Asthma", "Arthritis",
    "Chronic Migraine", "Anemia", "Hypothyroidism", "GERD",
    "Sleep Apnea", "Osteoporosis",
]
_MEDICATIONS = [
    "Metformin 500mg", "Lisinopril 10mg", "Albuterol 90mcg",
    "Levothyroxine 50mcg", "Omeprazole 20mg", "Atorvastatin 40mg",
    "Amlodipine 5mg", "Sertraline 50mg",
]
_LOG_EVENTS = [
    ("ERROR", "logged in from"),
    ("INFO", "support contact:"),
    ("WARN", "call us at"),
    ("INFO", "backup contact"),
    ("ERROR", "reset password"),
    ("INFO", "emergency:"),
    ("WARN", "user {} connected"),
    ("INFO", "hotline:"),
]
_LOG_IPS = ["192.168.1.1", "10.0.0.5", "172.16.0.3", "192.168.0.100"]


def _rand_email(first: str, last: str) -> str:
    domain = random.choice(_EMAIL_DOMAINS)
    sep = random.choice([".", "_", ""])
    return f"{first.lower()}{sep}{last.lower()}@{domain}"


def _rand_phone() -> str:
    area = random.randint(200, 999)
    mid = random.randint(100, 999)
    end = random.randint(1000, 9999)
    fmt = random.choice(["paren", "dot", "intl"])
    if fmt == "paren":
        return f"({area}) {mid}-{end}"
    elif fmt == "dot":
        return f"{area}.{mid}.{end}"
    else:
        return f"+1-{area}-{mid}-{end}"


def _rand_ssn() -> str:
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"


def _rand_insurance(patient_id: int) -> str:
    suffix = "".join(random.choices(string.ascii_uppercase, k=3))
    return f"INS-{patient_id}-{suffix}"


def _rand_dob() -> str:
    year = random.randint(1950, 2000)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def _generate_task1() -> dict:
    """Generate task1: random emails and phone numbers in server_logs.txt."""
    people = random.sample(
        [(f, l) for f in _FIRST_NAMES for l in _LAST_NAMES], 4
    )
    emails = [_rand_email(f, l) for f, l in people]
    phones = [_rand_phone() for _ in range(4)]
    ip = random.choice(_LOG_IPS)

    dates = ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02",
             "2026-01-03", "2026-01-03", "2026-01-04", "2026-01-04"]
    log_lines = [
        f"{dates[0]} ERROR user {emails[0]} logged in from {ip}",
        f"{dates[1]} INFO support contact: {emails[1]}",
        f"{dates[2]} WARN call us at {phones[0]} for help",
        f"{dates[3]} INFO backup contact {phones[1]}",
        f"{dates[4]} ERROR {emails[2]} reset password",
        f"{dates[5]} INFO emergency: {phones[2]}",
        f"{dates[6]} WARN user {emails[3]} connected",
        f"{dates[7]} INFO hotline: {phones[3]}",
    ]
    content = "\n".join(log_lines) + "\n"

    return {
        "description": (
            "TASK 1 (Easy): Redact all PII from server_logs.txt. "
            "Find and redact: 4 email addresses and 4 phone numbers. "
            "Use list_files, read_file, redact_text tools, then submit."
        ),
        "files": {"server_logs.txt": content},
        "pii_list": emails + phones,
        "safe_list": [ip],
        "max_steps": 25,
    }


def _generate_task2() -> dict:
    """Generate task2: random names and SSNs in customers.csv."""
    people = random.sample(
        [(f, l) for f in _FIRST_NAMES for l in _LAST_NAMES], 4
    )
    colors = random.sample(_COLORS, 4)
    rows = []
    pii_list = []
    safe_list = []
    for i, ((first, last), color) in enumerate(zip(people, colors), start=1001):
        name = f"{first} {last}"
        ssn = _rand_ssn()
        email = _rand_email(first, last)
        rows.append(f"{i},{name},{ssn},{email},{color}")
        pii_list.extend([name, ssn])
        safe_list.extend([str(i), email, color])

    content = "user_id,name,ssn,email,favorite_color\n" + "\n".join(rows) + "\n"

    return {
        "description": (
            "TASK 2 (Medium): Redact PII from customers.csv. "
            "Redact the 'name' and 'ssn' columns ONLY. "
            "Do NOT touch user_id, email, or favorite_color. "
            "CSV structure must remain valid after redaction."
        ),
        "files": {"customers.csv": content},
        "pii_list": pii_list,
        "safe_list": safe_list,
        "max_steps": 20,
    }


def _generate_task3() -> dict:
    """Generate task3: random patient breach with cross-file redaction."""
    # Generate 3 patients; one random target
    all_ids = random.sample(range(1000, 9999), 3)
    target_id = all_ids[1]  # middle one is always the target

    patients = []
    for pid in all_ids:
        first = random.choice(_FIRST_NAMES)
        last = random.choice(_LAST_NAMES)
        diag = random.choice(_DIAGNOSES)
        if pid == target_id:
            patients.append({
                "id": pid,
                "name": f"{first} {last}",
                "diagnosis": diag,
                "medication": random.choice(_MEDICATIONS),
                "ssn": _rand_ssn(),
                "phone": _rand_phone(),
                "email": _rand_email(first, last),
                "insurance_id": _rand_insurance(pid),
                "date_of_birth": _rand_dob(),
            })
        else:
            patients.append({"id": pid, "name": f"{first} {last}", "diagnosis": diag})

    target = next(p for p in patients if p["id"] == target_id)
    decoys = [p for p in patients if p["id"] != target_id]

    breach_content = (
        "SECURITY INCIDENT REPORT\n"
        f"Date: 2026-04-{random.randint(1,8):02d}\n"
        "Severity: HIGH\n"
        f"Patient ID: {target_id} data was compromised in the breach.\n"
        "Immediate action required.\n"
    )

    pii_list = [
        target["name"], target["diagnosis"], target["medication"],
        target["ssn"], target["phone"], target["email"],
        target["insurance_id"], target["date_of_birth"],
    ]
    safe_list = []
    for d in decoys:
        safe_list.extend([d["name"], d["diagnosis"]])

    return {
        "description": (
            "TASK 3 (Hard): Cross-file PII redaction. "
            "Step 1: Read breach_report.txt to find the compromised Patient ID. "
            "Step 2: Read medical_database.json to locate that patient. "
            "Step 3: Redact ALL sensitive fields of the compromised patient: "
            "name, diagnosis, medication, ssn, phone, email, insurance_id, date_of_birth. "
            "Do NOT touch any other patients. JSON must remain valid after redaction."
        ),
        "files": {
            "breach_report.txt": breach_content,
            "medical_database.json": json.dumps(patients, indent=2),
        },
        "target_patient_id": target_id,
        "pii_list": pii_list,
        "safe_list": safe_list,
        "max_steps": 20,
    }


class DataPrivacyEnvironment(Environment):
    """
    DataPrivacyEnv — PII Redaction RL Environment.
    Agent learns to find and redact sensitive data across 3 tasks.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    _VALID_TASKS = {"task1_plaintext", "task2_csv", "task3_json"}
    _GENERATORS = {
        "task1_plaintext": _generate_task1,
        "task2_csv": _generate_task2,
        "task3_json": _generate_task3,
    }

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_id = "task1_plaintext"
        self._task_data: dict = {}
        self.virtual_fs = {}
        self.cumulative_reward = 0.0
        self.found_patient_id = False
        self.done = False

    def reset(self, task_id: str = "task1_plaintext") -> DataPrivacyObservation:
        task_id = task_id if task_id in self._VALID_TASKS else "task1_plaintext"
        self.task_id = task_id
        self._task_data = self._GENERATORS[task_id]()
        task = self._task_data
        self.virtual_fs = {k: v for k, v in task["files"].items()}
        self.cumulative_reward = 0.0
        self.found_patient_id = False
        self.done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return DataPrivacyObservation(
            task_id=self.task_id,
            task_description=task["description"],
            available_tools=["list_files", "read_file", "redact_text", "submit"],
            last_action_result="Environment reset. Start with list_files or read_file.",
            last_reward=0.0,
            cumulative_reward=0.0,
            files_in_scope=list(self.virtual_fs.keys()),
            step_number=0,
            max_steps=task["max_steps"],
            done=False,
            reward=0.0,
        )

    def step(self, action: DataPrivacyAction) -> DataPrivacyObservation:  # type: ignore[override]
        self._state.step_count += 1
        task = self._task_data
        reward = 0.0
        result_msg = ""
        done = False

        # Parse the JSON message from agent
        try:
            parsed = json.loads(action.message)
            tool = parsed.get("tool", "")
        except (json.JSONDecodeError, AttributeError):
            return DataPrivacyObservation(
                task_id=self.task_id,
                task_description=task["description"],
                available_tools=["list_files", "read_file", "redact_text", "submit"],
                last_action_result="Error: Send valid JSON with a 'tool' field.",
                last_reward=-0.05,
                cumulative_reward=self.cumulative_reward,
                files_in_scope=list(self.virtual_fs.keys()),
                step_number=self._state.step_count,
                max_steps=task["max_steps"],
                done=False,
                reward=-0.05,
            )

        # Max steps check
        if self._state.step_count >= task["max_steps"]:
            done = True
            result_msg = "Max steps reached. Episode ending."

        elif tool == "list_files":
            result_msg = f"Files available: {list(self.virtual_fs.keys())}"
            reward = 0.0

        elif tool == "read_file":
            file_path = parsed.get("file_path", "")
            if file_path not in self.virtual_fs:
                result_msg = f"Error: '{file_path}' not found. Available: {list(self.virtual_fs.keys())}"
                reward = -0.05
            else:
                content = self.virtual_fs[file_path]
                result_msg = f"=== {file_path} ===\n{content}"
                reward = 0.05  # small reward for reading a file (ensures min score > 0)
                # Task 3 extra credit for reading breach report
                if (self.task_id == "task3_json" and
                        file_path == "breach_report.txt" and
                        not self.found_patient_id):
                    self.found_patient_id = True
                    reward = 0.15
                    result_msg += "\n[HINT] Find Patient ID in this file, then check medical_database.json"

        elif tool == "redact_text":
            file_path = parsed.get("file_path", "")
            target = parsed.get("target_string", "")
            replacement = parsed.get("replacement", "[REDACTED]")

            if file_path not in self.virtual_fs:
                result_msg = f"Error: '{file_path}' not found."
                reward = -0.05
            elif not target:
                result_msg = "Error: 'target_string' required."
                reward = -0.05
            elif target not in self.virtual_fs[file_path]:
                result_msg = f"'{target}' not found in {file_path}."
                reward = -0.05
            else:
                old_content = self.virtual_fs[file_path]
                new_content = old_content.replace(target, replacement)
                self.virtual_fs[file_path] = new_content

                pii_list = task.get("pii_list", [])
                safe_list = task.get("safe_list", [])

                if target in pii_list:
                    reward = +0.2
                    result_msg = f"Correctly redacted PII: '{target}'"
                elif target in safe_list:
                    reward = -0.2
                    result_msg = f"False positive — '{target}' is NOT PII!"
                else:
                    reward = -0.1
                    result_msg = f"'{target}' redacted but not in known PII list."

                # CSV structure validation
                if (self.task_id == "task2_csv" and
                        "customers.csv" in self.virtual_fs):
                    try:
                        list(csv.reader(
                            io.StringIO(self.virtual_fs["customers.csv"])
                        ))
                    except csv.Error:
                        reward = -1.0
                        done = True
                        result_msg = "CRITICAL: CSV structure broken! Episode failed."

                # JSON structure validation
                if (self.task_id == "task3_json" and
                        "medical_database.json" in self.virtual_fs):
                    try:
                        json.loads(self.virtual_fs["medical_database.json"])
                    except json.JSONDecodeError:
                        reward = -1.0
                        done = True
                        result_msg = "CRITICAL: JSON structure broken! Episode failed."

        elif tool == "submit":
            done = True
            penalty = 0.0
            pii_list = task.get("pii_list", [])

            # Penalize missed PII
            for pii in pii_list:
                for content in self.virtual_fs.values():
                    if pii in content:
                        penalty += 0.3
                        break

            # Task 3 — check correct patient redacted
            if self.task_id == "task3_json":
                try:
                    target_id = self._task_data.get("target_patient_id")
                    data = json.loads(self.virtual_fs["medical_database.json"])
                    patient = next((p for p in data if p["id"] == target_id), None)
                    if patient and patient.get("diagnosis") != "[REDACTED]":
                        penalty += 0.4
                    # Penalize wrong patients touched
                    for p in data:
                        if p["id"] != target_id and p.get("diagnosis") == "[REDACTED]":
                            penalty += 0.5
                except json.JSONDecodeError:
                    penalty += 1.0

            reward = -penalty
            result_msg = (
                f"Episode complete. "
                f"Penalty for missed PII: -{penalty:.2f}. "
                f"Total cumulative: {self.cumulative_reward + reward:.2f}"
            )
        else:
            result_msg = f"Unknown tool '{tool}'. Use: list_files, read_file, redact_text, submit"
            reward = -0.05

        self.cumulative_reward += reward
        # Clamp to (0.01, 1.99) so normalized score = cumulative/2.0 is always
        # strictly in (0.005, 0.995) — satisfies validator requirement of (0, 1) exclusive.
        self.cumulative_reward = max(0.01, min(1.99, self.cumulative_reward))
        self.done = done

        return DataPrivacyObservation(
            task_id=self.task_id,
            task_description=task["description"],
            available_tools=["list_files", "read_file", "redact_text", "submit"],
            last_action_result=result_msg,
            last_reward=round(reward, 3),
            cumulative_reward=round(self.cumulative_reward, 3),
            files_in_scope=list(self.virtual_fs.keys()),
            step_number=self._state.step_count,
            max_steps=task["max_steps"],
            done=done,
            reward=round(reward, 3),
        )

    @property
    def state(self) -> State:
        return self._state
