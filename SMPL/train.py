import torch
import torch.nn as nn
import torch.optim as optim

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true.to(device), y_pred.to(device)).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_model(model, labeled_train_loader, unlabeled_train_loader_augmented, optimizer, criterion, device):
    SEED = 2
    EPOCHS = 100
    BETA_ZERO = 8
    a = torch.tensor(8,dtype=torch.float32)
    i = torch.tensor(1,dtype=torch.float32)

    torch.manual_seed(SEED)
    labeled_batches = iter(labeled_train_loader)
    torch.manual_seed(SEED)
    unlabeled_batches = iter(unlabeled_train_loader)
    torch.manual_seed(SEED)
    unlabeled_batches_augmented = iter(unlabeled_train_loader_augmented)

    r_seed = 2
    print(device)
    for epoch_num in range(EPOCHS):
        acc = 0
        print("\n\n\n\n\nEPOCH NUMBER", epoch_num + 1, "\n\n\n\n")
        print("Loss_1 , Loss_old, Loss_new, Loss_partial, Loss_UDA, Loss_MPL\n")
        model.train()
        loss_per_epoch_1 = 0

        for epoch in range(len(unlabeled_train_loader)):
            try:
                labeled_images, labels = next(labeled_batches)
            except StopIteration:
                r_seed += 1
                torch.manual_seed(r_seed)
                labeled_batches = iter(labeled_train_loader)
                labeled_images, labels = next(labeled_batches)

            try:
                unlabeled_images, uy = next(unlabeled_batches)
            except StopIteration:
                torch.manual_seed(r_seed)
                unlabeled_batches = iter(unlabeled_train_loader)
                unlabeled_images, uy = next(unlabeled_batches)

            try:
                unlabeled_images_augmented, uya = next(unlabeled_batches_augmented)
            except StopIteration:
                torch.manual_seed(r_seed)
                unlabeled_batches_augmented = iter(unlabeled_train_loader_augmented)
                unlabeled_images_augmented, uya = next(unlabeled_batches_augmented)

            with torch.no_grad():
                pseudo_labels = torch.argmax(torch.softmax(model(unlabeled_images.to(device)), dim=1), dim=1).to(device)

            optimizer.zero_grad()

            unlabeled_logits_augmented = model(unlabeled_images_augmented.to(device)).to(device)
            unlabeled_logits = model(unlabeled_images.to(device)).to(device)
            labeled_logits = model(labeled_images.to(device)).to(device)

            loss_1 = nn.CrossEntropyLoss()(unlabeled_logits_augmented, pseudo_labels)
            loss_old = nn.CrossEntropyLoss()(labeled_logits.to(device), labels.to(device)).to(device)

            (loss_1).backward()
            optimizer.step()

            new_labeled_logits = model(labeled_images.to(device)).to(device)
            loss_ce_y = nn.CrossEntropyLoss()(new_labeled_logits, labels.to(device)).to(device)

            new_unlabeled_logits = model(unlabeled_images.to(device)).to(device)
            new_unlabeled_logits_augmented = model(unlabeled_images_augmented.to(device)).to(device)

            with torch.no_grad():
                pseudo_labels_unlabeled = torch.softmax(model(unlabeled_images.to(device)).to(device), dim=1)

            BETA_K = BETA_ZERO * torch.min(i, ((epoch + 1) / a))

            mask, ____ = torch.max(torch.softmax(new_unlabeled_logits_augmented, dim=1), dim=1)
            mask = mask.ge(0.95).float()

            loss_partial = BETA_K * torch.mean(
                torch.sum(-(torch.softmax(new_unlabeled_logits, dim=1)) * torch.log(
                    torch.softmax(new_unlabeled_logits_augmented, dim=1)), dim=-1) * mask)
            loss_uda = loss_ce_y + loss_partial

            with torch.no_grad():
                new_pseudo_labels = torch.argmax(torch.softmax(new_unlabeled_logits, dim=1), dim=1).to(device)

            ce_new = loss_ce_y.item()
            ce_old = loss_old.item()
            change = ce_new - ce_old
            loss_mpl = ((change) * (nn.CrossEntropyLoss()(new_unlabeled_logits, new_pseudo_labels)))

            print(loss_1.item(), loss_old.item(), loss_ce_y.clone().item(), loss_partial.item(),
                  loss_uda.item(), loss_mpl.item())

            loss_2 = loss_uda + loss_mpl
            (loss_2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            optimizer.step()

            loss_per_epoch_1 += loss_ce_y.item()

        model.eval()

        for _ in range(len(test_loader)):
            try:
                test_images, test_labels = next(test_batches)
            except StopIteration:
                test_batches = iter(test_loader)
                test_images, test_labels = next(test_batches)

            test_logits = model(test_images.to(device))
            pred_labels = torch.argmax(torch.softmax(test_logits, dim=1).to(device), dim=1).to(device)
            acc += accuracy_fn(test_labels, pred_labels)

        print("Accuracy per epoch :", acc / len(test_loader))
        print(f"Loss_1:{loss_per_epoch_1/len(unlabeled_train_loader)} ")
