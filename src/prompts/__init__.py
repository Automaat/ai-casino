"""Prompt loading utilities."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


class PromptDirectoryNotFoundError(ValueError):
    """Prompt directory not found."""


class PromptNotFoundError(FileNotFoundError):
    """Prompt file not found."""


class PromptVariableMissingError(ValueError):
    """Required variable missing in prompt template."""


class PromptLoader:
    """Loads and renders prompt templates."""

    def __init__(self, agent_name: str) -> None:
        """Initialize prompt loader for an agent.

        Args:
            agent_name: Name of the agent (subdirectory in prompts/)
        """
        self.agent_dir = PROMPTS_DIR / agent_name
        if not self.agent_dir.exists():
            msg = f"Prompt directory not found: {self.agent_dir}"
            raise PromptDirectoryNotFoundError(msg)

    def load(self, prompt_name: str, **kwargs: str) -> str:
        """Load and render prompt with f-string style variables.

        Args:
            prompt_name: Filename without .txt extension
            **kwargs: Variables to interpolate into template

        Returns:
            Rendered prompt string

        Raises:
            PromptNotFoundError: If prompt file doesn't exist
            PromptVariableMissingError: If required variable is missing
        """
        prompt_path = self.agent_dir / f"{prompt_name}.txt"

        if not prompt_path.exists():
            msg = f"Prompt not found: {prompt_path}"
            raise PromptNotFoundError(msg)

        template = prompt_path.read_text(encoding="utf-8")

        try:
            return template.format(**kwargs)
        except KeyError as e:
            msg = f"Missing variable {e} in prompt {prompt_name}"
            raise PromptVariableMissingError(msg) from e


def load_prompt(agent_name: str, prompt_name: str, **kwargs: str) -> str:
    """Convenience function to load and render a prompt.

    Args:
        agent_name: Name of the agent
        prompt_name: Filename without .txt extension
        **kwargs: Variables to interpolate

    Returns:
        Rendered prompt string
    """
    return PromptLoader(agent_name).load(prompt_name, **kwargs)
