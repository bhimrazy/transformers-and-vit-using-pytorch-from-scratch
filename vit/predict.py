import os
import timm
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

k = 5

CLASSSES_PATH = os.path.join(BASE_DIR, "classes.txt")
imagenet_labels = dict(enumerate(open(CLASSSES_PATH)))

# model = torch.load("model.pth")
# model.eval()

model_name = "vit_base_patch16_384"
model = timm.create_model(model_name, pretrained=True)
model.eval()

img = (np.array(Image.open("cat.png")) / 128) - 1  # in the range -1, 1


def predict(path):
    if not os.path.isfile(path):
        raise FileNotFoundError

    print("Predicting...\n")
    inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    logits = model(inp)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_ixs = probs[0].topk(k)

    print("Classes Probabilities.")
    for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
        ix = ix_.item()
        prob = prob_.item()
        cls = imagenet_labels[ix].strip()
        print(f"{i}: {cls:<45} --- {prob:.4f}")


if __name__ == "__main__":
    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--path', '-i', type=str, help="image path")
    args = parser.parse_args()
    # print(vars(args))
    predict(args.path)
