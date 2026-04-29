from agentkit.utils.logging import setup_logging


def test_setup_logging() -> None:
    # Test just runs the setup to cover the lines. It uses loguru, so we just want to ensure it doesn't crash
    setup_logging(debug=True)
    setup_logging(debug=False)
