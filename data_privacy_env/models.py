# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class ListFilesAction(Action):
    tool: Literal["list_files"] = "list_files"
    directory: str = Field(default=".", description="Directory to list")


class ReadFileAction(Action):
    tool: Literal["read_file"] = "read_file"
    file_path: str = Field(..., description="Path of file to read")


class RedactTextAction(Action):
    tool: Literal["redact_text"] = "redact_text"
    file_path: str = Field(..., description="File to redact from")
    target_string: str = Field(..., description="Exact PII string to redact")
    replacement: str = Field(default="[REDACTED]", description="Replacement text")


class SubmitAction(Action):
    tool: Literal["submit"] = "submit"


# Use a wrapper action since OpenEnv needs single Action class
class DataPrivacyAction(Action):
    """Wrapper action - agent sends JSON with 'tool' field"""
    message: str = Field(..., description="""
    Send a JSON string with one of these tools:

    SCAN phase tools:
    {"tool": "list_files", "directory": "."}
    {"tool": "read_file", "file_path": "server_logs.txt"}
    {"tool": "flag_candidate", "text": "john@email.com", "file_path": "server_logs.txt", "pii_type": "EMAIL"}
    {"tool": "advance_phase"}

    CLASSIFY phase tools:
    {"tool": "list_candidates"}
    {"tool": "classify_candidate", "candidate_id": "c0", "confirmed": true}
    {"tool": "advance_phase"}

    REDACT phase tools:
    {"tool": "redact_span", "candidate_id": "c0"}
    {"tool": "submit"}
    """)


class DataPrivacyObservation(Observation):
    task_id: str = Field(default="", description="Current task ID")
    task_description: str = Field(default="", description="What the agent must do")
    available_tools: list[str] = Field(
        default=["list_files", "read_file", "flag_candidate", "advance_phase"],
        description="Tools you can use in the current phase"
    )
    last_action_result: str = Field(default="", description="Result of last action")
    last_reward: float = Field(default=0.0, description="Reward from last step")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")
    files_in_scope: list[str] = Field(default=[], description="Files available")
    step_number: int = Field(default=0, description="Current step")
    max_steps: int = Field(default=25, description="Max steps allowed")
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.0, description="Step or terminal reward")
    agent_phase: str = Field(default="SCAN", description="Current phase: SCAN|CLASSIFY|REDACT")
    curriculum_level: int = Field(default=1, description="Difficulty level 1-4")
    candidate_count: int = Field(default=0, description="Number of flagged candidates")
    classified_count: int = Field(default=0, description="Number of confirmed PII candidates")
    last_candidate_id: str | None = Field(default=None, description="ID of last flagged candidate")
    metrics: dict = Field(default_factory=dict, description="Episode metrics (scan_recall, precision, redact_completeness)")
