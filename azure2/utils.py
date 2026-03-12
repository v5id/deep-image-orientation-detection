import torch
from sklearn.metrics import accuracy_score


def evaluate(model, dataloader, device):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for images, target in dataloader:

            images = images.to(device)

            outputs = model(images)

            pred = outputs.argmax(dim=1).cpu().numpy()

            preds.extend(pred)
            labels.extend(target.numpy())

    return accuracy_score(labels, preds)