import os
import logging

def setup_logging(log_dir: str = "outputs/logs", level: int = logging.INFO):
    """Configura logging a fichero en log_dir."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")
    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()

def ensure_dir(path: str):
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)