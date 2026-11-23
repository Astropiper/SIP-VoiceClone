"""Write report figures without requiring a display server."""

from pathlib import Path
from tempfile import NamedTemporaryFile


def save_png(figure, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        FigureCanvasAgg(figure).print_png(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
