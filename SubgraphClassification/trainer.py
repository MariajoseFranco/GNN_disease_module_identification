from evaluator import evaluating_model


def training_loop(model, g, labels, train_idx, val_idx, optimizer, loss_fn, epochs, device):
    """
    Trains the GNN model for link prediction using positive and negative edges.

    Args:
        model (nn.Module): The GNN model.
        predictor (nn.Module): The predictor module that computes edge scores.
        train_pos_u (Tensor): Source nodes of positive training edges.
        train_pos_v (Tensor): Destination nodes of positive training edges.
        train_neg_u (Tensor): Source nodes of negative training edges.
        train_neg_v (Tensor): Destination nodes of negative training edges.
        g (DGLGraph): The homogeneous protein-protein interaction graph.
        features (Tensor): Node feature matrix.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (torch.nn.Module): Loss function (e.g., BCEWithLogitsLoss).

    Returns:
        None: Trains the model in-place.
    """
    model.train()

    g = g.to(device)
    features = g.ndata['feat'].to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    losses = []
    val_accuracy = []
    val_f1s = []
    val_precisions = []
    val_recalls = []

    best_f1 = -1
    best_model_state = None
    best_threshold = 0.5

    for epoch in range(epochs):
        logits = model(g, features)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, acc, f1, prec, rec, auc, _, current_thresh = evaluating_model(
            model, g, labels, val_idx, device
        )
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict()
            best_threshold = current_thresh
        val_accuracy.append(acc)
        val_f1s.append(f1)
        val_precisions.append(prec)
        val_recalls.append(rec)
        if epoch % 5 == 0:
            print(
                f"Epoch {epoch}, Loss {loss.item():.4f}"
                f", Val Acc {acc:.4f}, Val F1 {f1:.4f}"
                f", Val Prec {prec:.4f}, Val Rec {rec:.4f}, Val AUC Score {auc:.4f}"
            )
    print(f"Best Threshold: {best_threshold:.2f}")
    model.load_state_dict(best_model_state)
    return losses, val_accuracy, val_f1s, val_precisions, val_recalls, best_threshold
