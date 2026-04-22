"""
pi-ai provider base module.

Re-exports ProviderInterface from registry for convenience.
"""

from pi_ai.registry import ProviderInterface, ProviderProtocol

__all__ = ["ProviderInterface", "ProviderProtocol"]