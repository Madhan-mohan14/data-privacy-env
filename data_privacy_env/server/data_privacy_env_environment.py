# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import csv
import io
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataPrivacyAction, DataPrivacyObservation
except ImportError:
    from models import DataPrivacyAction, DataPrivacyObservation

TASKS = {
    "task1_plaintext": {
        "description": (
            "TASK 1 (Easy): Redact all PII from server_logs.txt. "
            "Find and redact: 4 email addresses and 4 phone numbers. "
            "Use list_files, read_file, redact_text tools, then submit."
        ),
        "files": {
            "server_logs.txt": (
                "2026-01-01 ERROR user john.doe@company.com logged in from 192.168.1.1\n"
                "2026-01-01 INFO support contact: jane.smith@email.com\n"
                "2026-01-02 WARN call us at (555) 123-4567 for help\n"
                "2026-01-02 INFO backup contact 555.987.6543\n"
                "2026-01-03 ERROR admin@internal.org reset password\n"
                "2026-01-03 INFO emergency: +1-800-555-0199\n"
                "2026-01-04 WARN user bob.wilson@domain.co.uk connected\n"
                "2026-01-04 INFO hotline: (800) 555-9876\n"
            )
        },
        "pii_list": [
            "john.doe@company.com",
            "jane.smith@email.com",
            "(555) 123-4567",
            "555.987.6543",
            "admin@internal.org",
            "+1-800-555-0199",
            "bob.wilson@domain.co.uk",
            "(800) 555-9876",
        ],
        "safe_list": ["192.168.1.1"],
        "max_steps": 25,
    },
    "task2_csv": {
        "description": (
            "TASK 2 (Medium): Redact PII from customers.csv. "
            "Redact the 'name' and 'ssn' columns ONLY. "
            "Do NOT touch user_id, email, or favorite_color. "
            "CSV structure must remain valid after redaction."
        ),
        "files": {
            "customers.csv": (
                "user_id,name,ssn,email,favorite_color\n"
                "1001,Alice Johnson,123-45-6789,alice@example.com,blue\n"
                "1002,Bob Martinez,987-65-4321,bob@example.com,green\n"
                "1003,Carol White,456-78-9012,carol@example.com,red\n"
                "1004,David Brown,321-54-9876,david@example.com,purple\n"
            )
        },
        "pii_list": [
            "Alice Johnson", "123-45-6789",
            "Bob Martinez", "987-65-4321",
            "Carol White", "456-78-9012",
            "David Brown", "321-54-9876",
        ],
        "safe_list": [
            "1001", "1002", "1003", "1004",
            "alice@example.com", "bob@example.com",
            "carol@example.com", "david@example.com",
            "blue", "green", "red", "purple",
        ],
        "max_steps": 20,
    },
    "task3_json": {
        "description": (
            "TASK 3 (Hard): Cross-file PII redaction. "
            "Step 1: Read breach_report.txt to find the compromised Patient ID. "
            "Step 2: Read medical_database.json to locate that patient. "
            "Step 3: Redact ALL sensitive fields of the compromised patient: "
            "name, diagnosis, medication, ssn, phone, email, insurance_id, date_of_birth. "
            "Do NOT touch any other patients. JSON must remain valid after redaction."
        ),
        "files": {
            "breach_report.txt": (
                "SECURITY INCIDENT REPORT\n"
                "Date: 2026-04-06\n"
                "Severity: HIGH\n"
                "Patient ID: 9982 data was compromised in the breach.\n"
                "Immediate action required.\n"
            ),
            "medical_database.json": json.dumps([
                {"id": 9981, "name": "John Smith", "diagnosis": "Hypertension"},
                {
                    "id": 9982,
                    "name": "Jane Doe",
                    "diagnosis": "Type 2 Diabetes",
                    "medication": "Metformin 500mg",
                    "ssn": "456-78-9012",
                    "phone": "555-234-5678",
                    "email": "jane.doe@private.com",
                    "insurance_id": "INS-9982-XYZ",
                    "date_of_birth": "1985-03-15",
                },
                {"id": 9983, "name": "Bob Johnson", "diagnosis": "Asthma"},
            ], indent=2),
        },
        "target_patient_id": 9982,
        "pii_list": [
            "Jane Doe",
            "Type 2 Diabetes",
            "Metformin 500mg",
            "456-78-9012",
            "555-234-5678",
            "jane.doe@private.com",
            "INS-9982-XYZ",
            "1985-03-15",
        ],
        "safe_list": ["John Smith", "Hypertension", "Bob Johnson", "Asthma"],
        "max_steps": 20,
    },
}


class DataPrivacyEnvironment(Environment):
    """
    DataPrivacyEnv — PII Redaction RL Environment.
    Agent learns to find and redact sensitive data across 3 tasks.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_id = "task1_plaintext"
        self.virtual_fs = {}
        self.cumulative_reward = 0.0
        self.found_patient_id = False
        self.done = False

    def reset(self, task_id: str = "task1_plaintext") -> DataPrivacyObservation:
        task_id = task_id if task_id in TASKS else "task1_plaintext"
        self.task_id = task_id
        task = TASKS[task_id]
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
        task = TASKS[self.task_id]
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
                    data = json.loads(self.virtual_fs["medical_database.json"])
                    patient = next((p for p in data if p["id"] == 9982), None)
                    if patient and patient.get("diagnosis") != "[REDACTED]":
                        penalty += 0.4
                    # Penalize wrong patients touched
                    for p in data:
                        if p["id"] != 9982 and p.get("diagnosis") == "[REDACTED]":
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
