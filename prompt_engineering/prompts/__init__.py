from .few_shot import build_prompt as few_shot_prompt
from .role_prompt import build_prompt as role_prompt
from .structured_prompt import build_prompt as structured_prompt
from .zero_shot import build_prompt as zero_shot_prompt

__all__ = [
    "zero_shot_prompt",
    "role_prompt",
    "few_shot_prompt",
    "structured_prompt",
]
