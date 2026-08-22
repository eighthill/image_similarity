import torch
from torchvision import models
from torchvision import transforms


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Embedding device: {device}")

model = models.resnet50(weights="DEFAULT")
model.fc = torch.nn.Identity()
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


import time
from concurrent.futures import ThreadPoolExecutor

def resize_image(image):
    return image.resize((224, 224))

def extract_embeddings(images):

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=8) as executor:
        resized = list(
            executor.map(
                resize_image,
                images,
            )
        )

    resize_time = time.perf_counter() - start

    start = time.perf_counter()

    tensors = torch.stack([
        transforms.functional.pil_to_tensor(image).float() / 255.0
        for image in resized
    ])

    tensor_time = time.perf_counter() - start

    start = time.perf_counter()

    tensors = tensors.to(device)

    transfer_time = time.perf_counter() - start

    start = time.perf_counter()

    with torch.inference_mode():
        embeddings = model(tensors)

    model_time = time.perf_counter() - start

    start = time.perf_counter()

    embeddings = embeddings.cpu().numpy()

    output_time = time.perf_counter() - start

    print(
        f"Embedding: "
        f"Resize {resize_time:.2f}s | "
        f"Tensor {tensor_time:.2f}s | "
        f"Transfer {transfer_time:.2f}s | "
        f"Model {model_time:.2f}s | "
        f"Output {output_time:.2f}s"
    )

    return embeddings