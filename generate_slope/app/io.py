from pathlib import Path
import laspy


def load_las(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS not found: {path}")
    return laspy.read(path)


def save_las(las_data, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_data.write(out_path)
