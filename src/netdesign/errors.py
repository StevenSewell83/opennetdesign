"""Exception hierarchy for netdesign."""

__all__ = ["ConfigError", "DataError", "NetDesignError", "PhysicsError"]


class NetDesignError(Exception):
    """Base class for all netdesign errors."""


class ConfigError(NetDesignError):
    """Bad YAML or configuration values. The message names the config path."""


class DataError(NetDesignError):
    """Bad or unreadable input files."""


class PhysicsError(NetDesignError):
    """Physically invalid model (e.g. Vp/Vs below sqrt(4/3), negative velocity)."""
