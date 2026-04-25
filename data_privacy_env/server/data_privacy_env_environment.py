# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataPrivacyAction, DataPrivacyObservation
    from ..curriculum import CurriculumManager, generate_task_for_level
except ImportError:
    from models import DataPrivacyAction, DataPrivacyObservation
    from curriculum import CurriculumManager, generate_task_for_level


PHASE_TOOLS: dict[str, list[str]] = {
    "SCAN": ["list_files", "read_file", "flag_candidate", "advance_phase"],
    "CLASSIFY": ["list_candidates", "classify_candidate", "advance_phase"],
    "REDACT": ["redact_span", "submit"],
}

_curriculum = CurriculumManager()


class ComplianceGuardEnv(Environment):
    """
    ComplianceGuard — 3-phase PII redaction RL environment.

    Phases: SCAN → CLASSIFY → REDACT
    Phase gate enforced via raise ValueError (caught by TRL natively + HTTP try/except).
    Near-binary reward: 1.0 perfect / 0.3+0.6*product partial / 0.05 floor.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS: int = 30

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.phase: str = "SCAN"
        self.level: int = 1
        self.virtual_fs: dict[str, str] = {}
        self.pii_list: list[str] = []
        self.candidates: dict[str, dict] = {}  # cid -> {text, file_path, pii_type, confirmed}
        self._next_cid: int = 0
        self.done: bool = False
        self.reward: float = 0.0
        self.cumulative_reward: float = 0.0
        self._task_description: str = ""

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, level: int | None = None, **kwargs) -> DataPrivacyObservation:
        if seed is not None:
            random.seed(seed)

        self.level = level if level is not None else _curriculum.get_level()
        files, pii_list = generate_task_for_level(self.level)

        self.virtual_fs = dict(files)
        self.pii_list = list(pii_list)
        self.candidates = {}
        self._next_cid = 0
        self.phase = "SCAN"
        self.done = False
        self.reward = 0.0
        self.cumulative_reward = 0.0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_description = (
            f"[Level {self.level}] Redact all PII from {len(self.virtual_fs)} file(s). "
            "Phase 1 SCAN: read files and flag PII candidates. "
            "Phase 2 CLASSIFY: confirm or reject each candidate. "
            "Phase 3 REDACT: redact confirmed candidates and submit."
        )

        return self._initial_obs()

    def step(self, action: DataPrivacyAction, **kwargs) -> DataPrivacyObservation:
        self._state.step_count += 1

        # Enforce step limit — OpenEnv requires reward strictly in (0, 1), never 0.0
        if self._state.step_count >= self.MAX_STEPS and not self.done:
            self.reward, _ = self._compute_reward()
            self.done = True
            _curriculum.record_episode(self.reward)
            return self._make_obs(
                self.reward,
                f"Max steps ({self.MAX_STEPS}) reached. Episode terminated.",
                done=True,
            )

        try:
            parsed = json.loads(action.message)
            tool = parsed.get("tool", "")
        except (json.JSONDecodeError, AttributeError):
            return self._make_obs(-0.05, "Error: send valid JSON with a 'tool' field.")

        method = getattr(self, f"_tool_{tool}", None)
        if method is None:
            return self._make_obs(-0.05, f"Unknown tool '{tool}'. Allowed in {self.phase}: {PHASE_TOOLS[self.phase]}")

        try:
            result = method(parsed)
        except ValueError as e:
            return self._make_obs(-0.05, str(e))

        if self.done:
            step_reward = self.reward
            _curriculum.record_episode(step_reward)
            return self._make_obs(step_reward, result, done=True)

        return self._make_obs(0.0, result)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Phase gate
    # ------------------------------------------------------------------

    def _require_phase(self, required: str, tool_name: str) -> None:
        if self.phase != required:
            raise ValueError(
                f"'{tool_name}' is only allowed in {required} phase. "
                f"Currently in {self.phase}. Allowed tools: {PHASE_TOOLS[self.phase]}"
            )

    # ------------------------------------------------------------------
    # SCAN phase tools
    # ------------------------------------------------------------------

    def _tool_list_files(self, parsed: dict) -> str:
        self._require_phase("SCAN", "list_files")
        return f"Files: {list(self.virtual_fs.keys())}"

    def _tool_read_file(self, parsed: dict) -> str:
        self._require_phase("SCAN", "read_file")
        fp = parsed.get("file_path", "")
        if fp not in self.virtual_fs:
            raise ValueError(f"'{fp}' not found. Available: {list(self.virtual_fs.keys())}")
        return f"=== {fp} ===\n{self.virtual_fs[fp]}"

    def _tool_flag_candidate(self, parsed: dict) -> str:
        self._require_phase("SCAN", "flag_candidate")
        text = parsed.get("text", "").strip()
        file_path = parsed.get("file_path", "")
        pii_type = parsed.get("pii_type", "OTHER")

        if not text:
            raise ValueError("'text' field required for flag_candidate.")
        if file_path and file_path not in self.virtual_fs:
            raise ValueError(f"'{file_path}' not in scope.")

        existing = [cid for cid, c in self.candidates.items()
                    if c["text"].strip() == text.strip()]
        if existing:
            return f"Already flagged as {existing[0]}. Use that candidate_id instead."

        cid = f"c{self._next_cid}"
        self._next_cid += 1
        self.candidates[cid] = {
            "text": text,
            "file_path": file_path,
            "pii_type": pii_type,
            "confirmed": None,
            "redacted": False,
        }
        return f"Flagged {cid}: {pii_type} | {text!r}"

    def _tool_advance_phase(self, parsed: dict) -> str:
        if self.phase == "SCAN":
            if not self.candidates:
                raise ValueError(
                    "No candidates flagged. Use flag_candidate before advancing."
                )
            self.phase = "CLASSIFY"
            return (
                f"Advanced to CLASSIFY. {len(self.candidates)} candidate(s) to review. "
                f"Use list_candidates then classify_candidate for each."
            )
        elif self.phase == "CLASSIFY":
            unclassified = [c for c, v in self.candidates.items() if v["confirmed"] is None]
            if unclassified:
                raise ValueError(
                    f"Classify all candidates first. Unclassified: {unclassified}"
                )
            self.phase = "REDACT"
            confirmed = [c for c, v in self.candidates.items() if v["confirmed"]]
            return (
                f"Advanced to REDACT. {len(confirmed)} confirmed PII candidate(s). "
                "Use redact_span for each, then submit."
            )
        else:
            raise ValueError("Already in REDACT phase. Use redact_span and submit.")

    # ------------------------------------------------------------------
    # CLASSIFY phase tools
    # ------------------------------------------------------------------

    def _tool_list_candidates(self, parsed: dict) -> str:
        self._require_phase("CLASSIFY", "list_candidates")
        if not self.candidates:
            return "No candidates flagged."
        lines = []
        for cid, info in self.candidates.items():
            status = {True: "CONFIRMED", False: "REJECTED", None: "PENDING"}[info["confirmed"]]
            lines.append(f"  {cid}: [{status}] {info['pii_type']} | {info['text']!r}")
        return "Candidates:\n" + "\n".join(lines)

    def _tool_classify_candidate(self, parsed: dict) -> str:
        self._require_phase("CLASSIFY", "classify_candidate")
        cid = parsed.get("candidate_id", "")
        if cid not in self.candidates:
            raise ValueError(f"Unknown candidate_id '{cid}'. Use list_candidates.")
        confirmed = parsed.get("confirmed")
        if confirmed is None:
            raise ValueError("'confirmed' field required (true or false).")
        self.candidates[cid]["confirmed"] = bool(confirmed)
        status = "CONFIRMED" if confirmed else "REJECTED"
        return f"{cid} {status}: {self.candidates[cid]['text']!r}"

    # ------------------------------------------------------------------
    # REDACT phase tools
    # ------------------------------------------------------------------

    def _tool_redact_span(self, parsed: dict) -> str:
        self._require_phase("REDACT", "redact_span")
        cid = parsed.get("candidate_id", "")
        if cid not in self.candidates:
            raise ValueError(f"Unknown candidate_id '{cid}'.")
        c = self.candidates[cid]
        if not c["confirmed"]:
            raise ValueError(f"'{cid}' was not confirmed as PII. Only redact confirmed candidates.")
        if c["redacted"]:
            return f"{cid} already redacted."

        text = c["text"]
        fp = c["file_path"]
        if fp and fp in self.virtual_fs and text in self.virtual_fs[fp]:
            self.virtual_fs[fp] = self.virtual_fs[fp].replace(text, "[REDACTED]")
        else:
            # Search all files
            for fname in self.virtual_fs:
                if text in self.virtual_fs[fname]:
                    self.virtual_fs[fname] = self.virtual_fs[fname].replace(text, "[REDACTED]")
        c["redacted"] = True
        return f"Redacted {cid}: {text!r} → [REDACTED]"

    def _tool_submit(self, parsed: dict) -> str:
        self._require_phase("REDACT", "submit")
        self.reward, metrics = self._compute_reward()
        self.done = True
        return (
            f"Episode complete. reward={self.reward:.4f} | "
            f"scan_recall={metrics.get('scan_recall', 0):.2f} "
            f"precision={metrics.get('precision', 0):.2f} "
            f"redact_completeness={metrics.get('redact_completeness', 0):.2f}"
        )

    # ------------------------------------------------------------------
    # Reward computation (near-binary)
    # ------------------------------------------------------------------

    def _compute_reward(self) -> tuple[float, dict]:
        total_pii = len(self.pii_list)
        if total_pii == 0:
            return 0.5, {}

        # scan_recall: fraction of real PII that was flagged as a candidate
        flagged_texts = {v["text"] for v in self.candidates.values()}
        scanned_hits = sum(1 for pii in self.pii_list if pii in flagged_texts or
                           any(pii in ft or ft in pii for ft in flagged_texts))
        scan_recall = scanned_hits / total_pii

        # precision: among confirmed candidates, fraction that are real PII
        confirmed = [v for v in self.candidates.values() if v["confirmed"]]
        if not confirmed:
            precision = 0.0
        else:
            true_positives = sum(
                1 for c in confirmed
                if any(c["text"] in pii or pii in c["text"] for pii in self.pii_list)
            )
            precision = true_positives / max(1, len(confirmed))

        # redact_completeness: fraction of real PII actually removed from files
        all_content = "\n".join(self.virtual_fs.values())
        still_present = sum(1 for pii in self.pii_list if pii in all_content)
        redact_completeness = 1.0 - (still_present / total_pii)

        if scan_recall >= 0.99 and precision >= 0.99 and redact_completeness >= 0.99:
            reward = 1.0
        elif scan_recall >= 0.5 and redact_completeness > 0:
            reward = 0.3 + 0.6 * (scan_recall * precision * redact_completeness)
        else:
            reward = 0.05

        reward = max(0.001, min(0.999, reward))
        metrics = {
            "scan_recall": round(scan_recall, 4),
            "precision": round(precision, 4),
            "redact_completeness": round(redact_completeness, 4),
        }
        return reward, metrics

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _make_obs(
        self,
        reward: float,
        result: str,
        done: bool = False,
    ) -> DataPrivacyObservation:
        self.cumulative_reward += reward
        confirmed = [c for c, v in self.candidates.items() if v["confirmed"]]
        last_cid = list(self.candidates.keys())[-1] if self.candidates else None
        _, metrics = self._compute_reward() if done else (0.0, {})

        return DataPrivacyObservation(
            task_id=f"level_{self.level}",
            task_description=self._task_description,
            available_tools=PHASE_TOOLS[self.phase],
            last_action_result=result,
            last_reward=round(reward, 4),
            cumulative_reward=round(self.cumulative_reward, 4),
            files_in_scope=list(self.virtual_fs.keys()),
            step_number=self._state.step_count,
            max_steps=self.MAX_STEPS,
            done=done,
            reward=round(reward, 4),
            agent_phase=self.phase,
            curriculum_level=self.level,
            candidate_count=len(self.candidates),
            classified_count=len(confirmed),
            last_candidate_id=last_cid,
            metrics=metrics,
        )

    def _initial_obs(self) -> DataPrivacyObservation:
        return DataPrivacyObservation(
            task_id=f"level_{self.level}",
            task_description=self._task_description,
            available_tools=PHASE_TOOLS["SCAN"],
            last_action_result=(
                f"Environment reset. Level {self.level}. "
                f"Files: {list(self.virtual_fs.keys())}. "
                "Start with list_files then read_file to scan for PII."
            ),
            last_reward=0.001,
            cumulative_reward=0.0,
            files_in_scope=list(self.virtual_fs.keys()),
            step_number=0,
            max_steps=self.MAX_STEPS,
            done=False,
            reward=0.001,  # OpenEnv requires reward strictly > 0.0 on all observations
            agent_phase="SCAN",
            curriculum_level=self.level,
            candidate_count=0,
            classified_count=0,
            last_candidate_id=None,
            metrics={},
        )
