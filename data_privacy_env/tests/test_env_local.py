"""Local env tests — no server required. Run: pytest tests/test_env_local.py -v"""
import json
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DataPrivacyAction
from server.data_privacy_env_environment import ComplianceGuardEnv
from curriculum.generators import generate_task_for_level


def _act(tool: str, **kwargs) -> DataPrivacyAction:
    return DataPrivacyAction(message=json.dumps({"tool": tool, **kwargs}))


@pytest.fixture
def env():
    e = ComplianceGuardEnv()
    e.reset(seed=42, level=1)
    return e


# ---- Phase gate -------------------------------------------------------

def test_phase_gate_scan_blocks_classify(env):
    obs = env.step(_act("classify_candidate", candidate_id="c0", confirmed=True))
    assert obs.agent_phase == "SCAN"
    assert "SCAN" in obs.last_action_result or "only allowed" in obs.last_action_result
    assert obs.reward <= 0


def test_phase_gate_scan_blocks_redact(env):
    obs = env.step(_act("redact_span", candidate_id="c0"))
    assert "only allowed" in obs.last_action_result


# ---- Full episode flow -----------------------------------------------

def test_full_l1_episode_reaches_nonzero_reward():
    """L1 seed=42: SCAN all files, CLASSIFY all, REDACT all, SUBMIT → reward > floor."""
    e = ComplianceGuardEnv()
    obs = e.reset(seed=42, level=1)
    assert obs.agent_phase == "SCAN"
    assert obs.curriculum_level == 1

    # SCAN: list and read
    e.step(_act("list_files"))
    for fname in e.virtual_fs:
        e.step(_act("read_file", file_path=fname))

    # Flag all real PII
    for pii in e.pii_list:
        for fname, content in e.virtual_fs.items():
            if pii in content:
                e.step(_act("flag_candidate", text=pii, file_path=fname, pii_type="OTHER"))
                break

    # Advance to CLASSIFY
    obs = e.step(_act("advance_phase"))
    assert obs.agent_phase == "CLASSIFY"

    # Confirm all
    for cid in list(e.candidates.keys()):
        e.step(_act("classify_candidate", candidate_id=cid, confirmed=True))

    # Advance to REDACT
    obs = e.step(_act("advance_phase"))
    assert obs.agent_phase == "REDACT"

    # Redact all confirmed
    for cid, info in e.candidates.items():
        if info["confirmed"]:
            e.step(_act("redact_span", candidate_id=cid))

    # Submit
    obs = e.step(_act("submit"))
    assert obs.done
    assert obs.reward > 0.05, f"Expected reward > floor 0.05, got {obs.reward}"
    assert 0.001 <= obs.reward <= 0.999


# ---- Reward boundary -------------------------------------------------

def test_perfect_episode_gives_near_one_reward():
    """Flagging + confirming + redacting all PII exactly should give reward close to 1.0."""
    e = ComplianceGuardEnv()
    e.reset(seed=7, level=1)

    for pii in e.pii_list:
        for fname, content in e.virtual_fs.items():
            if pii in content:
                e.step(_act("flag_candidate", text=pii, file_path=fname, pii_type="OTHER"))
                break

    e.step(_act("advance_phase"))
    for cid in list(e.candidates.keys()):
        e.step(_act("classify_candidate", candidate_id=cid, confirmed=True))
    e.step(_act("advance_phase"))
    for cid, info in e.candidates.items():
        if info["confirmed"]:
            e.step(_act("redact_span", candidate_id=cid))
    obs = e.step(_act("submit"))

    assert obs.reward >= 0.85, f"Perfect episode should score ≥ 0.85, got {obs.reward}"


# ---- Dispatcher correctness ------------------------------------------

def test_unknown_tool_returns_error_not_exception(env):
    obs = env.step(_act("nonexistent_tool"))
    assert "Unknown tool" in obs.last_action_result
    assert obs.reward < 0
    assert not obs.done


def test_advance_without_candidates_raises_error(env):
    obs = env.step(_act("advance_phase"))
    assert "No candidates" in obs.last_action_result
    assert obs.agent_phase == "SCAN"  # should stay in SCAN


def test_observation_fields_populated(env):
    obs = env.step(_act("list_files"))
    assert obs.task_id == f"level_{env.level}"
    assert obs.agent_phase == "SCAN"
    assert obs.curriculum_level == 1
    assert isinstance(obs.metrics, dict)
    assert obs.available_tools == ["list_files", "read_file", "flag_candidate", "advance_phase"]


# ---- Fix 1: L3 pii_list must match file content ----------------------

def test_l3_pii_list_matches_file_content():
    """After Fix 1: every L3 pii_list item must be literally present in some file."""
    files, pii_list = generate_task_for_level(3)
    all_content = " ".join(files.values())
    for pii in pii_list:
        assert pii in all_content, f"PII '{pii}' not found in any L3 file content"
    has_obfuscated = any("[at]" in p or "dash" in p or "dot" in p for p in pii_list)
    assert has_obfuscated, "L3 pii_list should contain at least one obfuscated item"


def test_l4_red_herrings_not_in_pii_list():
    """L4 red herrings must appear in files but NOT in pii_list."""
    files, pii_list = generate_task_for_level(4)
    all_content = " ".join(files.values())
    assert "test@example.com" in all_content or "noreply@system.local" in all_content, \
        "L4 files should contain red herring addresses"
    for pii in pii_list:
        assert "test@example.com" not in pii, "Red herring test@example.com must not be in pii_list"
        assert "noreply@system.local" not in pii, "Red herring noreply@system.local must not be in pii_list"


def test_l3_perfect_episode_reward():
    """After Fix 1: agent flagging all L3 pii_list items exactly must get reward > 0.9."""
    e = ComplianceGuardEnv()
    e.reset(seed=42, level=3)

    for pii in e.pii_list:
        for fname, content in e.virtual_fs.items():
            if pii in content:
                e.step(_act("flag_candidate", text=pii, file_path=fname, pii_type="OTHER"))
                break

    obs = e.step(_act("advance_phase"))
    assert obs.agent_phase == "CLASSIFY", f"Expected CLASSIFY, got {obs.agent_phase}: {obs.last_action_result}"

    for cid in list(e.candidates.keys()):
        e.step(_act("classify_candidate", candidate_id=cid, confirmed=True))

    e.step(_act("advance_phase"))

    for cid, info in e.candidates.items():
        if info["confirmed"]:
            e.step(_act("redact_span", candidate_id=cid))

    obs = e.step(_act("submit"))
    assert obs.done
    assert obs.reward >= 0.9, f"Perfect L3 episode should score ≥ 0.9, got {obs.reward}"


# ---- Fix 2: timeout reward must be in (0, 1), initial obs reward != 0 ----

def test_timeout_reward_never_zero():
    """After Fix 2: episode timeout must return reward in [0.001, 0.999], not 0.0."""
    e = ComplianceGuardEnv()
    e.reset(level=1, seed=42)
    obs = None
    for _ in range(35):
        obs = e.step(_act("list_files"))
        if obs.done:
            break
    assert obs is not None
    assert obs.done is True, "Episode should have terminated by step limit"
    assert obs.reward >= 0.001, f"Timeout reward must be >= 0.001, got {obs.reward}"
    assert obs.reward <= 0.999, f"Timeout reward must be <= 0.999, got {obs.reward}"


def test_initial_obs_reward_not_zero():
    """After Fix 2: initial observation reward must not be exactly 0.0."""
    e = ComplianceGuardEnv()
    obs = e.reset(level=1, seed=42)
    assert obs.reward != 0.0, "Initial observation reward must not be exactly 0.0"
    assert obs.reward >= 0.001, f"Initial observation reward must be >= 0.001, got {obs.reward}"


# ---- Fix 3: duplicate flag prevention --------------------------------

def test_duplicate_flag_prevented():
    env = ComplianceGuardEnv()
    env.reset(level=1, seed=42)
    env.step(_act("list_files"))
    env.step(_act("read_file", file_path=list(env.virtual_fs.keys())[0]))
    pii = env.pii_list[0]
    env.step(_act("flag_candidate", text=pii, suspected_type="name"))
    # Try to flag the same text again with a different suspected type
    obs = env.step(_act("flag_candidate", text=pii, suspected_type="email"))
    assert len(env.candidates) == 1, "Duplicate flag should be prevented"
    assert "already" in obs.last_action_result.lower()
    print("PASS: duplicate flagging prevented")
