import logging
from datetime import datetime
from pathlib import Path

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{datetime.now():%d-%m-%Y_%H-%M-%S}.log"

logging.basicConfig(
    filename=log_file,
    format="[ %(asctime)s ] %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    encoding="utf-8",
)

Logger = logging.getLogger("HandwrittenNotesRAG")
