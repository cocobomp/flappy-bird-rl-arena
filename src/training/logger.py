"""Metrics logging for RL training experiments.

Provides episode-level metric collection, recent history retrieval,
and aggregate statistics (best score, average score).
"""

from __future__ import annotations


class MetricsLogger:
    """Collects and provides access to per-episode training metrics.

    Each logged episode is stored as a dict with at minimum:
    episode, score, reward, steps.  Optional agent_info fields are
    merged into the same dict.
    """

    def __init__(self):
        self.history: list[dict] = []

    def log_episode(
        self,
        episode: int,
        score: int,
        reward: float,
        steps: int,
        agent_info: dict | None = None,
    ) -> None:
        """Record metrics for a single training episode.

        Args:
            episode: Episode index.
            score: Game score achieved (e.g. pipes passed).
            reward: Cumulative shaped reward for the episode.
            steps: Number of environment steps taken.
            agent_info: Optional dict of agent-specific metrics
                (e.g. epsilon, loss) to merge into the log entry.
        """
        entry = {
            "episode": episode,
            "score": score,
            "reward": reward,
            "steps": steps,
        }
        if agent_info is not None:
            entry.update(agent_info)
        self.history.append(entry)

    def get_recent(self, n: int = 10) -> list[dict]:
        """Return the most recent *n* log entries.

        Args:
            n: Number of recent entries to return (default 10).

        Returns:
            A list of the last *n* episode dicts (or fewer if the
            history is shorter than *n*).
        """
        return self.history[-n:]

    @property
    def best_score(self) -> int:
        """Return the highest score across all logged episodes.

        Returns:
            The maximum score, or 0 if no episodes have been logged.
        """
        if not self.history:
            return 0
        return max(entry["score"] for entry in self.history)

    def average_score(self, n: int = 100) -> float:
        """Return the mean score over the last *n* episodes.

        Args:
            n: Number of recent episodes to average over (default 100).

        Returns:
            The average score, or 0.0 if no episodes have been logged.
        """
        if not self.history:
            return 0.0
        recent = self.history[-n:]
        return sum(entry["score"] for entry in recent) / len(recent)
