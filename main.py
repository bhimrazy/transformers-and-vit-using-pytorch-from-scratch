import click
from vit.predict import predict


@click.group()
def cli():
    pass


@cli.command()
@click.option('--path', help='Image path')
def vit_classify(path: str):
    """ViT Imagenet Classifier
    Args:
        path (str): image path
    """
    if not path:
        raise TypeError
    predict(path)


if __name__ == '__main__':
    cli()
