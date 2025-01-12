import torch
from .data import corrupt_mnist
from .model import MyAwesomeModel
import typer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(modelpath: str = "model.pt") -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(modelpath)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(f"models/{modelpath}", weights_only=True))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
