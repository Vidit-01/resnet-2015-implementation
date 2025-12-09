import torch
import time
from tqdm.notebook import tqdm
import importlib.util

def load_model(model_path, num_classes):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResNet(num_classes=num_classes)

def train_epoch(model,optimizer,criterion,train_loader,device):
    model.train()
    start_time = time.time()
    progress = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        leave=False
    )
    total, correct, total_loss = 0, 0, 0
    for batch_idx,(x, y) in progress:
        x, y = x.to(device), y.to(device)
        # print(x.shape)
        # print(x.type())
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        elapsed = time.time() - start_time
        batches_done = batch_idx + 1
        batches_total = len(train_loader)
        eta = elapsed / batches_done * (batches_total - batches_done)

        progress.set_postfix({
            "batch": f"{batches_done}/{batches_total}",
            "loss": f"{loss.item():.4f}",
            "eta": f"{eta:.1f}s"
        })
    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return correct / total