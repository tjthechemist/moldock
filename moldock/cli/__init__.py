"""MolDock CLI package."""

def main():
    from moldock.cli.main import main as _main
    return _main()

__all__ = ["main"]

