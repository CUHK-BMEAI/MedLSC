"""Compatibility with current and legacy adapter checkpoint keys."""

def get_adapter_config(config: dict) -> dict:
    """Read current or legacy adapter configuration without changing its contents."""
    return config["medlsc_cfg"] if "medlsc_cfg" in config else config["mslora_cfg"]


def is_adapter_weight_key(key: str) -> bool:
    """Recognize current and legacy adapter-weight checkpoint names."""
    return "medlsc_weight" in key or "mslora_weight" in key

