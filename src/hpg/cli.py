"""hpg command-line interface."""

import typer

app = typer.Typer(help="Hybrid System-1/System-2 Procedural Graph PoC (RefundDesk).", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Hybrid System-1/System-2 Procedural Graph PoC (RefundDesk)."""


@app.command()
def version() -> None:
    """Print package version."""
    from importlib.metadata import version as v

    typer.echo(v("hpg"))


if __name__ == "__main__":
    app()
