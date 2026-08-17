"""DFA-EHOA components, kept separate from the baseline implementation."""

from .dfa_ehoa import DFAEHOA

# SFIG-EHOA is the descriptive name used in the manuscript; DFA-EHOA is kept
# as the stable API/configuration name for backward compatibility.
SFIGEHOA = DFAEHOA

__all__ = ["DFAEHOA", "SFIGEHOA"]
