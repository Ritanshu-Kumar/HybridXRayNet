import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

from train_model import HybridXRayNet


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "best_model.pth"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

app = Flask(__name__)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found at: {MODEL_PATH}\n"
        "Train the model first using train_model.py."
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_to_idx = checkpoint.get(
    "class_to_idx",
    {
        "NORMAL": 0,
        "PNEUMONIA": 1
    }
)

idx_to_class = {
    value: key
    for key, value in class_to_idx.items()
}


model = HybridXRayNet(
    num_classes=len(class_to_idx)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# GRAD-CAM
# ============================================================

activations = None
gradients = None


def forward_hook(module, inputs, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


model.attention.register_forward_hook(
    forward_hook
)

model.attention.register_full_backward_hook(
    backward_hook
)


def generate_gradcam(
    input_tensor,
    target_class
):

    global activations
    global gradients

    model.zero_grad(
        set_to_none=True
    )

    outputs = model(
        input_tensor
    )

    target_score = outputs[
        :,
        target_class
    ].sum()

    target_score.backward()

    if (
        activations is None
        or gradients is None
    ):
        return None

    # Average gradients across spatial dimensions
    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Weighted feature maps
    cam = (
        weights * activations
    ).sum(
        dim=1,
        keepdim=True
    )

    cam = F.relu(
        cam
    )

    # Resize heatmap to input resolution
    cam = F.interpolate(
        cam,
        size=input_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    cam = (
        cam[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    cam -= cam.min()

    cam /= (
        cam.max()
        + 1e-8
    )

    return cam


def create_heatmap(
    image,
    cam
):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    image_array = np.asarray(
        image.convert("RGB")
        .resize((224, 224))
    ).astype(
        np.float32
    ) / 255.0

    figure = plt.figure(
        figsize=(5, 5)
    )

    axis = figure.add_axes(
        [0, 0, 1, 1]
    )

    axis.imshow(
        image_array
    )

    axis.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    axis.axis("off")

    buffer = io.BytesIO()

    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(
        figure
    )

    buffer.seek(0)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "file" not in request.files:

        return jsonify({
            "error":
                "No image uploaded."
        }), 400

    uploaded_file = request.files[
        "file"
    ]

    try:

        image = Image.open(
            uploaded_file.stream
        ).convert("L")

    except Exception:

        return jsonify({
            "error":
                "Invalid image file."
        }), 400


    input_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(DEVICE)


    # -------------------------
    # Prediction
    # -------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )


    predicted_class = idx_to_class[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )


    # -------------------------
    # Grad-CAM
    # -------------------------

    cam = generate_gradcam(
        input_tensor,
        predicted_index
    )

    heatmap = None

    if cam is not None:

        heatmap = create_heatmap(
            image,
            cam
        )


    return jsonify({

        "prediction":
            predicted_class,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "device":
            str(DEVICE),

        "gradcam":
            (
                f"data:image/png;base64,{heatmap}"
                if heatmap
                else None
            )
    })


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )