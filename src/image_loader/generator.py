from pathlib import Path
from PIL import Image, UnidentifiedImageError

"""
def image_generator(folder):

    folder = Path(folder)

    for file in folder.rglob("*"):

        if file.name.startswith("._"):
            continue

        if file.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        ]:
            continue

        try:
            with Image.open(file) as source:
                source.load()
                image = source.convert("RGB")

            yield image, str(file)

        except Exception as e:
            print(
                f"Skipping invalid file: {file}"
            )
            print(
                f"  PIL error: {type(e).__name__}: {e}"
            )
            continue
"""

def image_generator(folder):

    folder = Path(folder)

    for file in folder.rglob("*"):

        if file.name.startswith("._"):
            continue

        if file.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        ]:
            continue

        yield file