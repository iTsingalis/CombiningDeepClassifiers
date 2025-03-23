import torch
from tqdm import tqdm


def training(model, device, data_loader, optimizer, loss_fn, augmenter, reduction, log_softmax, softmax):
    model.train()
    train_loss, total_frame_acc, total_cnt = 0, 0, 0
    pbar = tqdm(data_loader, disable=False)
    for data in pbar:
        pbar.set_description("Training batch")

        batch_size = data[0].shape[0]

        if model.__class__.__name__ in ["Audio1DDevIdentification", "RawNet", "M5", "M11", "M18", "M34"]:
            inputs = data[0].reshape(batch_size, 1, -1).to(device)  # [batch, 3, 128, 1500]
        else:
            inputs = data[0].to(device)
        target_frame_labels = data[1].squeeze(1).to(device)  # ([1, ...,]), shape: [batch]

        if augmenter is not None:
            # The tensor has a shape like (tr_batch_size, num_channels, num_samples) and mode is per_example
            inputs = augmenter(
                samples=inputs.reshape(inputs.shape[0], 1, -1),
                sample_rate=22050).reshape(inputs.shape[0], -1)

        def closure(backward=True):
            if backward:
                optimizer.zero_grad()
            model_outputs = model(inputs)
            if softmax:
                cri_loss = loss_fn(torch.log(model_outputs), target_frame_labels)
            elif log_softmax:
                cri_loss = loss_fn(model_outputs, target_frame_labels)
            else:
                raise ValueError('We work with log softmax or softmax')
                cri_loss = loss_fn(model_outputs, target_frame_labels)

            create_graph = type(optimizer).__name__ == "AdaCubic" or type(
                optimizer).__name__ == "AdaHessian"
            if backward:
                cri_loss.backward(create_graph=create_graph)

            _, pred_label = torch.max(model_outputs.detach(), dim=1)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            return cri_loss, pred_label

        loss, pred_label = optimizer.step(closure=closure)

        if reduction:
            train_loss += loss.item() * batch_size  # Unscale the batch-averaged loss
        else:
            train_loss += loss.item()  # Unscale the batch-averaged loss

        acc = torch.sum((pred_label == target_frame_labels).float()).item()
        total_frame_acc += acc
        total_cnt += batch_size

    return train_loss / total_cnt, total_frame_acc / total_cnt


def validation(model, device, data_loader, loss_fn, reduction, log_softmax, softmax):
    valid_loss, total_frame_acc, total_cnt = 0, 0, 0

    pbar = tqdm(data_loader, disable=False)
    model.eval()
    with torch.no_grad():
        for data in pbar:
            pbar.set_description("Validation batch")
            batch_size = data[0].shape[0]

            if model.__class__.__name__ in ["Audio1DDevIdentification", "RawNet", "M5", "M11", "M18", "M34"]:
                inputs = data[0].reshape(batch_size, 1, -1).to(device)  # [batch, 3, 128, 1500]
            else:
                inputs = data[0].to(device)

            # inputs = data[0].reshape(batch_size, 1, -1).to(device)  # [batch, 3, 128, 1500]
            target_frame_labels = data[1].squeeze(1).to(device)
            # device_names = data[2]
            # audio_frame_indices = data[3]

            model_outputs = model(inputs)  # .squeeze()
            if loss_fn is not None:
                if softmax:
                    cri_loss = loss_fn(torch.log(model_outputs), target_frame_labels)
                elif log_softmax:
                    cri_loss = loss_fn(model_outputs, target_frame_labels)
                else:
                    raise ValueError('We work with log softmax or softmax')
                    cri_loss = loss_fn(model_outputs, target_frame_labels)

                if reduction:
                    valid_loss += cri_loss.item() * batch_size  # Unscale the batch-averaged loss
                else:
                    valid_loss += cri_loss.item()  # Unscale the batch-averaged loss

            _, pred_frame_labels = torch.max(model_outputs.data, 1)

            acc = torch.sum((pred_frame_labels == target_frame_labels).float()).item()
            total_frame_acc += acc
            total_cnt += batch_size

    return valid_loss / total_cnt, total_frame_acc / total_cnt
