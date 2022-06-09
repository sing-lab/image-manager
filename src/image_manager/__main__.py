"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Image_Manager."""


if __name__ == "__main__":
    main(prog_name="image_manager")  # pragma: no cover
