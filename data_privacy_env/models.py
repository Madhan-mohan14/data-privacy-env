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

    List files:
    {"tool": "list_files", "directory": "."}

    Read a file:
    {"tool": "read_file", "file_path": "server_logs.txt"}

    Redact PII:
    {"tool": "redact_text", "file_path": "server_logs.txt",
     "target_string": "john@email.com", "replacement": "[REDACTED]"}

    Submit when done:
    {"tool": "submit"}
    """)


class DataPrivacyObservation(Observation):
    task_id: str = Field(default="", description="Current task ID")
    task_description: str = Field(default="", description="What the agent must do")
    available_tools: list[str] = Field(
        default=["list_files", "read_file", "redact_text", "submit"],
        description="Tools you can use"
    )
    last_action_result: str = Field(default="", description="Result of last action")
    last_reward: float = Field(default=0.0, description="Reward from last step")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")
    files_in_scope: list[str] = Field(default=[], description="Files available")
    step_number: int = Field(default=0, description="Current step")
    max_steps: int = Field(default=25, description="Max steps allowed")
    done: bool = Field(default=False, description="Whether the episode has ended")
