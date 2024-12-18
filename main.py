import argparse
import os
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from data.dataset import ANP_Dataset, eval_AP_Dataset
from torch.nn.functional import normalize
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning Training")
    # Paths
    parser.add_argument("--augmented_file_path", type=str, required=True, help="Path to the processed dataset base")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Path to save models and logs")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--eval_epoch", type=int, default=1, help="Evaluate every n epochs")
    parser.add_argument("--save_epoch", type=int, default=10, help="Save model every n epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before stepping optimizer")

    # Model parameters
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ConvNeXt")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="contrastive_learning", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()
    return args

def get_model(pretrained=True):
    # ConvNeXt Small 모델 사용
    model = models.convnext_small(pretrained=pretrained)
    # 마지막 Linear 제거 -> 임베딩 추출용
    model.classifier[2] = nn.Identity()
    return model

def collate_fn_triplet(batch):
    # ANP_Dataset에서 anchor, positive, negative를 반환한다고 가정
    anchors = []
    positives = []
    negatives = []
    for b in batch:
        anchors.append(b['anchor'])
        positives.append(b['positive'])
        negatives.append(b['negative'])

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    return anchors, positives, negatives

def collate_fn_eval(batch):
    # AP_Dataset에서 anchor, image, label 을 반환한다고 가정
    anchors = []
    images = []
    labels = []
    for b in batch:
        anchors.append(b['anchor'])
        images.append(b['image'])
        labels.append(b['label'])
    anchors = torch.stack(anchors)
    images = torch.stack(images)
    return anchors, images, labels

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_emb, positive_emb, negative_emb):
        pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
        neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
        pos_loss = 0.5 * (pos_dist ** 2)
        neg_loss = 0.5 * torch.clamp(self.margin - neg_dist, min=0) ** 2
        loss = pos_loss.mean() + neg_loss.mean()
        return loss

def main():
    args = parse_args()

    accelerator = Accelerator()
    accelerator.init_trackers("contrastive_learning", config=vars(args))

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
        wandb.config.update(vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ANP_Dataset(augmented_file_path=args.augmented_file_path, mode='train', transform=transform)
    val_dataset = ANP_Dataset(augmented_file_path=args.augmented_file_path, mode='valid', transform=transform)
    eval_dataset = eval_AP_Dataset(data_path=args.augmented_file_path, transform=transform)

    # train_loader와 val_loader는 triplet (anchor, positive, negative) 반환
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_triplet, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_triplet, num_workers=args.num_workers)

    # eval_loader는 (anchor, image, label) 반환
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_eval, num_workers=args.num_workers)

    anchor_model = get_model(args.pretrained)
    posneg_model = get_model(args.pretrained)

    optimizer = optim.Adam(list(anchor_model.parameters()) + list(posneg_model.parameters()), lr=args.lr)
    contrastive_loss_fn = ContrastiveLoss(margin=1.0)

    anchor_model, posneg_model, optimizer, train_loader, val_loader, eval_loader = accelerator.prepare(
        anchor_model, posneg_model, optimizer, train_loader, val_loader, eval_loader
    )

    global_step = 0
    total_steps = len(train_loader) * args.epochs

    def get_embeddings_for_triplet(anchors, positives, negatives):
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

    def train_one_epoch(loader, pbar):
        nonlocal global_step
        anchor_model.train()
        posneg_model.train()
        total_loss = 0.0
        total_samples = 0

        optimizer.zero_grad(set_to_none=True)
        for step, (anchors, positives, negatives) in enumerate(loader):
            anchor_emb, positive_emb, negative_emb = get_embeddings_for_triplet(anchors, positives, negatives)
            loss = contrastive_loss_fn(anchor_emb, positive_emb, negative_emb)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            batch_size = anchors.size(0)
            total_loss += loss.item() * args.gradient_accumulation_steps * batch_size
            total_samples += batch_size

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.is_main_process:
                    global_step += 1
                    wandb.log({"step_loss": loss.item() * args.gradient_accumulation_steps})
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)

        avg_loss = total_loss / total_samples
        return avg_loss

    def validate_one_epoch(loader):
        # Validation loss 계산
        anchor_model.eval()
        posneg_model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for anchors, positives, negatives in loader:
                anchor_emb, positive_emb, negative_emb = get_embeddings_for_triplet(anchors, positives, negatives)
                loss = contrastive_loss_fn(anchor_emb, positive_emb, negative_emb)
                batch_size = anchors.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss

    def evaluate(loader, threshold=0.5):
        anchor_model.eval()
        posneg_model.eval()

        correct_normal = 0
        correct_fraud = 0
        total_normal = 0
        total_fraud = 0

        with torch.no_grad():
            for anchors, images, labels in loader:
                anchors = anchors.to(accelerator.device)
                images = images.to(accelerator.device)
                anchor_emb = anchor_model(anchors)
                image_emb = posneg_model(images)

                anchor_emb = normalize(anchor_emb, p=2, dim=1)
                image_emb = normalize(image_emb, p=2, dim=1)

                dist = torch.nn.functional.pairwise_distance(anchor_emb, image_emb)
                pred_positive = dist < threshold

                # lbl == "normal": positive 정답, lbl == "fraud": negative 정답
                for p, lbl in zip(pred_positive, labels):
                    if lbl == "normal":
                        total_normal += 1
                        # 정답이 normal이면 pred_positive가 True일 때 정답 맞춤
                        if p.item():
                            correct_normal += 1
                    elif lbl == "fraud":
                        total_fraud += 1
                        # 정답이 fraud이면 pred_positive가 False일 때 정답 맞춤
                        if not p.item():
                            correct_fraud += 1

        # 각 클래스별 정확도
        accuracy_normal = correct_normal / total_normal if total_normal > 0 else 0.0
        accuracy_fraud = correct_fraud / total_fraud if total_fraud > 0 else 0.0

        # 전체 정확도
        total = total_normal + total_fraud
        correct = correct_normal + correct_fraud
        accuracy = correct / total if total > 0 else 0.0

        # 필요하다면 wandb나 로그에 추가적으로 기록
        if accelerator.is_main_process:
            wandb.log({
                "accuracy_normal": accuracy_normal,
                "accuracy_fraud": accuracy_fraud,
                "overall_accuracy": accuracy,
                "total_samples": total,
                "total_normal": total_normal,
                "total_fraud": total_fraud,
                "correct_normal": correct_normal,
                "correct_fraud": correct_fraud
            })

        return accuracy

    pbar = None
    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, desc="Overall Training Progress", leave=True)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(train_loader, pbar)
        val_loss = validate_one_epoch(val_loader)  # validation loss 계산

        if (epoch + 1) % args.eval_epoch == 0:
            val_acc = evaluate(eval_loader)
            if accelerator.is_main_process:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc, "epoch": epoch+1})
        else:
            if accelerator.is_main_process:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1})

        if (epoch + 1) % args.save_epoch == 0:
            if accelerator.is_main_process:
                torch.save({
                    'epoch': epoch,
                    'anchor_model_state': anchor_model.state_dict(),
                    'posneg_model_state': posneg_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    if accelerator.is_main_process:
        pbar.close()
        wandb.finish()

if __name__ == "__main__":
    main()
