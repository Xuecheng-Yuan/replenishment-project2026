"""Minimal checks for the initial project scaffold."""


def test_package_import() -> None:
    import replenishment

    assert replenishment.__version__ == "0.1.0"

