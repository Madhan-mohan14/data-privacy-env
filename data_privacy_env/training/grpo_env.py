"""TRL-compatible wrapper: reset() returns str as TRL's environment_factory requires."""
import random

try:
    from ..server.data_privacy_env_environment import ComplianceGuardEnv
except ImportError:
    from server.data_privacy_env_environment import ComplianceGuardEnv


class ComplianceGuardEnvTRL(ComplianceGuardEnv):
    """
    Thin wrapper for TRL GRPOTrainer environment_factory.

    TRL expects reset() -> str and reads env.reward + env.done after step().
    ComplianceGuardEnv.reset() returns DataPrivacyObservation — this subclass
    overrides it to return the text content instead.
    """

    def reset(self, seed: int | None = None, level: int | None = None, **kwargs) -> str:  # type: ignore[override]
        if seed is not None:
            random.seed(seed)
        obs = super().reset(seed=seed, level=level, **kwargs)
        return obs.last_action_result
