# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data Privacy Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import DataPrivacyAction, DataPrivacyObservation
except ImportError:
    from models import DataPrivacyAction, DataPrivacyObservation


class DataPrivacyEnv(
    EnvClient[DataPrivacyAction, DataPrivacyObservation, State]
):
    """Client for DataPrivacyEnv — PII Redaction Environment."""

    def _step_payload(self, action: DataPrivacyAction) -> Dict:
        return {"message": action.message}

    def _parse_result(self, payload: Dict) -> StepResult[DataPrivacyObservation]:
        obs_data = payload.get("observation", {})
        observation = DataPrivacyObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            available_tools=obs_data.get("available_tools", []),
            last_action_result=obs_data.get("last_action_result", ""),
            last_reward=obs_data.get("last_reward", 0.0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            files_in_scope=obs_data.get("files_in_scope", []),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 25),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
