def evaluate_model(model, test_loader, criterion, device):
  acc=0
  torch.manual_seed(43)
  test_batches = iter(test_loader)
  torch.manual_seed(43)
  labeled_batches = iter(labeled_train_loader)
  model.eval()
  for _ in range(len(test_loader)):
    try:
        test_images, test_labels = next(test_batches)
    except StopIteration:
        test_batches = iter(test_loader)
        test_images, test_labels = next(test_batches)

    test_logits = model(test_images.to(device))
    pred_labels = torch.argmax(torch.softmax(test_logits,dim=1).to(device),dim=1).to(device)
    acc += accuracy_fn(test_labels, pred_labels)

  print("Accuracy per epoch :", acc/len(test_loader))