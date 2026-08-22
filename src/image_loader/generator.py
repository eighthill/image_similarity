from pathlib import Path
from PIL import Image, UnidentifiedImageError


def image_generator(folder):

    folder = Path(folder)

    for file in folder.rglob("*"):

        if file.name.startswith("._"):
            continue

        if file.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]:
            continue

        try:
            yield Image.open(file), str(file)

        except UnidentifiedImageError:
            print(
                f"Skipping invalid image: {file}"
            )