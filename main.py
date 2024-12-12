import argparse
import os
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from data.dataset import FD_Dataset
from torch.nn.functional import normalize

def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning Training")
    # Paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed dataset base")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to the raw dataset base")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Path to save models and logs")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate and save model every n epochs")

    # Model parameters
    parser.add_argument("--resnet_type", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Type of ResNet backbone")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="contrastive_learning", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()
    return args

def get_model(resnet_type="resnet18", pretrained=True):
    if resnet_type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif resnet_type == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Identity()
    return model

def collate_fn(batch):
    anchors = []
    positives = []
    negatives = []
    for b in batch:
        anchors.append(b['anchor'])     # Tensor
        positives.append(b['positive']) # Tensor
        negatives.append(b['negative']) # Tensor

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return anchors, positives, negatives

def main():
    args = parse_args()

    accelerator = Accelerator()
    # Initialize W&B
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
        wandb.config.update(vars(args))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = FD_Dataset(data_path=args.data_path, raw_data_path=args.raw_data_path, mode='train', transform=transform)
    val_dataset = FD_Dataset(data_path=args.data_path, raw_data_path=args.raw_data_path, mode='valid', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    anchor_model = get_model(args.resnet_type, args.pretrained)
    posneg_model = get_model(args.resnet_type, args.pretrained)

    optimizer = optim.Adam(list(anchor_model.parameters()) + list(posneg_model.parameters()), lr=args.lr)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

    anchor_model, posneg_model, optimizer, train_loader, val_loader = accelerator.prepare(
        anchor_model, posneg_model, optimizer, train_loader, val_loader
    )

    def get_embeddings(anchors, positives, negatives):
        # Now anchors, positives, negatives are all already Tensors on CPU
        # Move them to device
        anchors = anchors.to(accelerator.device)
        positives = positives.to(accelerator.device)
        negatives = negatives.to(accelerator.device)

        anchor_emb = anchor_model(anchors)
        positive_emb = posneg_model(positives)
        negative_emb = posneg_model(negatives)

        anchor_emb = normalize(anchor_emb, p=2, dim=1)
        positive_emb = normalize(positive_emb, p=2, dim=1)
        negative_emb = normalize(negative_emb, p=2, dim=1)

        return anchor_emb, positive_emb, negative_emb

    def run_epoch(loader, training=True):
        if training:
            anchor_model.train()
            posneg_model.train()
        else:
            anchor_model.eval()
            posneg_model.eval()

        total_loss = 0.0
        total_samples = 0

        for anchors, positives, negatives in loader:
            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                anchor_emb, positive_emb, negative_emb = get_embeddings(anchors, positives, negatives)
                loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)

                if training:
                    accelerator.backward(loss)
                    optimizer.step()

            total_loss += loss.item() * anchors.size(0)
            total_samples += anchors.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, training=True)
        if epoch % args.eval_every == 0:
            val_loss = run_epoch(val_loader, training=False)

            if accelerator.is_main_process:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
                torch.save({
                    'epoch': epoch,
                    'anchor_model_state': anchor_model.state_dict(),
                    'posneg_model_state': posneg_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"))
        else:
            if accelerator.is_main_process:
                wandb.log({"train_loss": train_loss, "epoch": epoch})

    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
