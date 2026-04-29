import sys
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

console = Console(theme=Theme({"info": "dim cyan", "warning": "yellow", "danger": "bold red"}))


def setup_logging(debug: bool = False) -> None:
    """
    Loguru'yu Rich kütüphanesiyle entegre ederek renkli ve düzenli loglama sağlar.
    Debug kapalıysa loguru uyarılara kadar susturulur.
    """
    # Mevcut tüm handler'ları kaldır
    logger.remove()

    if debug:
        log_level = "DEBUG"
        # Debug modunda Rich formatıyla konsola yaz
        logger.add(
            RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False),
            format="{message}",
            level=log_level,
            enqueue=True,
        )
    else:
        log_level = "WARNING"
        logger.add(sys.stderr, level=log_level)
