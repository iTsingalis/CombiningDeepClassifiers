import torch
from tqdm import tqdm


def training(model, device, data_loader, optimizer, loss_fn, reduction, log_softmax, softmax, verbose=False):
    model.train()
    train_loss, total_acc, total_cnt = 0, 0, 0

    pbar = tqdm(data_loader, disable=not verbose)
    for data in pbar:
        batch_size = data[0].shape[0]

        pbar.set_description("Training batch")
        inputs = data[0].to(device)
        target = data[1].squeeze(1).to(device)

        def closure(backward=True):
            if backward:
                optimizer.zero_grad()

            model_outputs = model(inputs)

            if softmax:
                cri_loss = loss_fn(torch.log(model_outputs), target)
            elif log_softmax:
                cri_loss = loss_fn(model_outputs, target)
            else:
                cri_loss = loss_fn(model_outputs, target)

            create_graph = type(optimizer).__name__ == "AdaCubic" or type(optimizer).__name__ == "AdaHessian"

            if backward:
                cri_loss.backward(create_graph=create_graph)

            _, pred_label = torch.max(model_outputs, dim=1)

            return cri_loss, pred_label

        loss, pred_label = optimizer.step(closure=closure)

        if reduction:
            train_loss += loss.item() * batch_size  # Unscale the batch-averaged loss
        else:
            train_loss += loss.item()

        acc = torch.sum((pred_label == target).float()).item()
        total_acc += acc

        total_cnt += batch_size

    return train_loss / total_cnt, total_acc / total_cnt


def validating(model, device, data_loader, loss_fn, reduction, log_softmax, softmax, verbose=False):
    valid_loss, total_acc, total_cnt = 0, 0, 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(data_loader, disable=not verbose)
        for data in pbar:
            batch_size = data[0].shape[0]

            pbar.set_description("Validation batch")

            inputs = data[0].to(device)
            target = data[1].to(device)

            model_outputs = model(inputs)

            if loss_fn is not None:
                if softmax:
                    loss = loss_fn(torch.log(model_outputs), target)
                elif log_softmax:
                    loss = loss_fn(model_outputs, target)
                else:
                    loss = loss_fn(model_outputs, target)

                if reduction:
                    valid_loss += loss.item() * batch_size  # Unscale the batch-averaged loss
                else:
                    valid_loss += loss.item()  # Unscale the batch-averaged loss

            _, pred_label = torch.max(model_outputs.data, 1)
            total_acc += torch.sum((pred_label == target).float()).item()

            total_cnt += batch_size

    return valid_loss / total_cnt, total_acc / total_cnt
