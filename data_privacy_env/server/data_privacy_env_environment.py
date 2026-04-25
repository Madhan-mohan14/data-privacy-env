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
    "SCAN":     ["list_files", "read_file", "flag_candidate", "advance_phase"],
    "CLASSIFY": ["list_candidates", "classify_candidate", "advance_phase"],
    "REDACT":   ["redact_span", "submit"],
}

_curriculum = CurriculumManager()


class ComplianceGuardEnv(Environment):
    """
    ComplianceGuard — 3-phase PII redaction RL environment.

    Phases: SCAN → CLASSIFY → REDACT

    Reward design (hackathon multi-component):
    ─────────────────────────────────────────
    Per-step (dense signal):
      flag_candidate real PII      → +0.04
      flag_candidate false positive→ -0.02
      classify_candidate correct   → +0.02
      classify_candidate wrong     → -0.03
      redact_span (confirmed)      → +0.03

    Terminal (at submit):
      Three independent components — harmonic mean prevents gaming any one:
        scan_f1           = 2*P*R / (P+R)   P/R over flagged vs real PII
        classify_accuracy = correct classifications / all classified
        redact_completeness = fraction of real PII removed from files
      reward = 0.05 + 0.949 * harmonic_mean(scan_f1, classify_acc, redact_complete)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 30

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.phase: str = "SCAN"
        self.level: int = 1
        self.virtual_fs: dict[str, str] = {}
        self.pii_list: list[str] = []
        self.candidates: dict[str, dict] = {}
        self._next_cid: int = 0
        self.done: bool = False
        self.reward: float = 0.0
        self.cumulative_reward: float = 0.0
        self._task_description: str = ""
        self._pending_step_reward: float = 0.0  # set by tool methods, read by step()

    # ── OpenEnv required interface ─────────────────────────────────────────

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
        self._pending_step_reward = 0.0
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
        self._pending_step_reward = 0.0

        if self._state.step_count >= self.MAX_STEPS and not self.done:
            self.reward, _ = self._compute_reward()
            self.done = True
            _curriculum.record_episode(self.reward)
            return self._make_obs(
                self.reward,
                f"Max steps ({self.MAX_STEPS}) reached. Episode terminated. reward={self.reward:.4f}",
                done=True,
            )

        try:
            parsed = json.loads(action.message)
            tool = parsed.get("tool", "")
        except (json.JSONDecodeError, AttributeError):
            return self._make_obs(-0.05, "Error: send valid JSON with a 'tool' field.")

        method = getattr(self, f"_tool_{tool}", None)
        if method is None:
            return self._make_obs(
                -0.05,
                f"Unknown tool '{tool}'. Allowed in {self.phase}: {PHASE_TOOLS[self.phase]}",
            )

        try:
            result = method(parsed)
        except ValueError as e:
            return self._make_obs(-0.05, str(e))

        step_reward = self._pending_step_reward

        if self.done:
            _curriculum.record_episode(self.reward)
            return self._make_obs(self.reward, result, done=True)

        return self._make_obs(step_reward, result)

    @property
    def state(self) -> State:
        return self._state

    # ── Phase gate ────────────────────────────────────────────────────────

    def _require_phase(self, required: str, tool_name: str) -> None:
        if self.phase != required:
            raise ValueError(
                f"'{tool_name}' is only allowed in {required} phase. "
                f"Currently in {self.phase}. Allowed: {PHASE_TOOLS[self.phase]}"
            )

    # ── SCAN tools ─────────────────────────────────────────────────────────

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

        existing = [cid for cid, c in self.candidates.items() if c["text"].strip() == text]
        if existing:
            return f"Already flagged as {existing[0]}."

        is_real_pii = any(text in pii or pii in text for pii in self.pii_list)
        self._pending_step_reward = 0.04 if is_real_pii else -0.02

        cid = f"c{self._next_cid}"
        self._next_cid += 1
        self.candidates[cid] = {
            "text": text,
            "file_path": file_path,
            "pii_type": pii_type,
            "confirmed": None,
            "redacted": False,
        }
        hint = " [real PII]" if is_real_pii else " [not in PII list — check carefully]"
        return f"Flagged {cid}: {pii_type} | {text!r}{hint}"

    def _tool_advance_phase(self, parsed: dict) -> str:
        if self.phase == "SCAN":
            if not self.candidates:
                raise ValueError("No candidates flagged. Use flag_candidate before advancing.")
            self.phase = "CLASSIFY"
            return (
                f"Advanced to CLASSIFY. {len(self.candidates)} candidate(s) to review. "
                "Use list_candidates then classify_candidate for each."
            )
        elif self.phase == "CLASSIFY":
            unclassified = [c for c, v in self.candidates.items() if v["confirmed"] is None]
            if unclassified:
                raise ValueError(f"Classify all candidates first. Unclassified: {unclassified}")
            self.phase = "REDACT"
            confirmed = [c for c, v in self.candidates.items() if v["confirmed"]]
            return (
                f"Advanced to REDACT. {len(confirmed)} confirmed PII candidate(s). "
                "Use redact_span for each confirmed candidate, then submit."
            )
        else:
            raise ValueError("Already in REDACT phase. Use redact_span and submit.")

    # ── CLASSIFY tools ────────────────────────────────────────────────────

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
        confirmed_input = parsed.get("confirmed")
        if confirmed_input is None:
            raise ValueError("'confirmed' field required (true or false).")

        confirmed = bool(confirmed_input)
        text = self.candidates[cid]["text"]
        is_real_pii = any(text in pii or pii in text for pii in self.pii_list)
        correct = confirmed == is_real_pii
        self._pending_step_reward = 0.02 if correct else -0.03

        self.candidates[cid]["confirmed"] = confirmed
        status = "CONFIRMED" if confirmed else "REJECTED"
        return f"{cid} {status}: {text!r}"

    # ── REDACT tools ──────────────────────────────────────────────────────

    def _tool_redact_span(self, parsed: dict) -> str:
        self._require_phase("REDACT", "redact_span")
        cid = parsed.get("candidate_id", "")
        if cid not in self.candidates:
            raise ValueError(f"Unknown candidate_id '{cid}'.")
        c = self.candidates[cid]
        if not c["confirmed"]:
            raise ValueError(f"'{cid}' was not confirmed. Only redact confirmed candidates.")
        if c["redacted"]:
            return f"{cid} already redacted."

        text = c["text"]
        fp = c["file_path"]
        if fp and fp in self.virtual_fs and text in self.virtual_fs[fp]:
            self.virtual_fs[fp] = self.virtual_fs[fp].replace(text, "[REDACTED]")
        else:
            for fname in self.virtual_fs:
                if text in self.virtual_fs[fname]:
                    self.virtual_fs[fname] = self.virtual_fs[fname].replace(text, "[REDACTED]")
        c["redacted"] = True
        self._pending_step_reward = 0.03
        return f"Redacted {cid}: {text!r} → [REDACTED]"

    def _tool_submit(self, parsed: dict) -> str:
        self._require_phase("REDACT", "submit")
        self.reward, metrics = self._compute_reward()
        self.done = True
        return (
            f"Episode complete. reward={self.reward:.4f} | "
            f"scan_f1={metrics.get('scan_f1', 0):.3f} "
            f"classify_acc={metrics.get('classify_accuracy', 0):.3f} "
            f"redact_complete={metrics.get('redact_completeness', 0):.3f} "
            f"harmonic={metrics.get('harmonic_mean', 0):.3f}"
        )

    # ── Reward computation (3-component, hackathon-grade) ─────────────────

    def _compute_reward(self) -> tuple[float, dict]:
        """
        Three independent reward components via harmonic mean.

        Using harmonic mean prevents an agent from gaming one component
        to compensate for failures in others (e.g. flagging everything
        for high recall but terrible precision).
        """
        total_pii = len(self.pii_list)
        if total_pii == 0:
            return 0.5, {}

        # ── Component 1: Scan F1 ───────────────────────────────────────────
        # Balances recall (finding all PII) with precision (not over-flagging)
        flagged_texts = {v["text"] for v in self.candidates.values()}
        tp_scan = sum(
            1 for pii in self.pii_list
            if pii in flagged_texts or any(pii in ft or ft in pii for ft in flagged_texts)
        )
        fp_scan = sum(
            1 for ft in flagged_texts
            if not any(ft in pii or pii in ft for pii in self.pii_list)
        )
        scan_recall = tp_scan / total_pii
        scan_precision = tp_scan / max(1, tp_scan + fp_scan)
        scan_f1 = (
            2.0 * scan_precision * scan_recall
            / max(1e-9, scan_precision + scan_recall)
        )

        # ── Component 2: Classification accuracy ──────────────────────────
        # How correctly the agent confirmed real PII and rejected non-PII
        classified = [v for v in self.candidates.values() if v["confirmed"] is not None]
        if classified:
            correct = sum(
                1 for v in classified
                if bool(v["confirmed"]) == any(
                    v["text"] in pii or pii in v["text"] for pii in self.pii_list
                )
            )
            classify_acc = correct / len(classified)
        else:
            classify_acc = 0.0

        # ── Component 3: Redact completeness ──────────────────────────────
        # Fraction of real PII actually removed from file contents
        all_content = "\n".join(self.virtual_fs.values())
        still_present = sum(1 for pii in self.pii_list if pii in all_content)
        redact_complete = 1.0 - (still_present / total_pii)

        # ── Harmonic mean of all three components ─────────────────────────
        components = [scan_f1, classify_acc, redact_complete]
        if all(c > 1e-9 for c in components):
            harmonic = len(components) / sum(1.0 / c for c in components)
        else:
            harmonic = 0.0

        # ── Smooth reward curve: 0.05 → 0.999 ─────────────────────────────
        if harmonic >= 0.99:
            reward = 0.999
        elif harmonic > 0:
            reward = 0.05 + 0.949 * harmonic
        else:
            reward = 0.05

        return max(0.001, min(0.999, reward)), {
            "scan_f1": round(scan_f1, 4),
            "classify_accuracy": round(classify_acc, 4),
            "redact_completeness": round(redact_complete, 4),
            "harmonic_mean": round(harmonic, 4),
        }

    # ── Observation helpers ───────────────────────────────────────────────

    def _make_obs(self, reward: float, result: str, done: bool = False) -> DataPrivacyObservation:
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
            reward=0.001,
            agent_phase="SCAN",
            curriculum_level=self.level,
            candidate_count=0,
            classified_count=0,
            last_candidate_id=None,
            metrics={},
        )
