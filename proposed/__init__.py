"""DFA-EHOA components, kept separate from the baseline implementation."""

from .dfa_ehoa import DFAEHOA
from .ecg_ehoa import ECGEHOA
from .memetic_ehoa import MemeticEHOA
from .aligned_ehoa import AlignedEHOA
from .guarded_ehoa import GuardedEHOA
from .portfolio_ehoa import PortfolioEHOA
from .safe_portfolio_ehoa import SafePortfolioEHOA

# SFIG-EHOA is the descriptive name used in the manuscript; DFA-EHOA is kept
# as the stable API/configuration name for backward compatibility.
SFIGEHOA = DFAEHOA

__all__ = ["DFAEHOA", "SFIGEHOA", "ECGEHOA", "MemeticEHOA", "AlignedEHOA", "GuardedEHOA", "PortfolioEHOA", "SafePortfolioEHOA"]
