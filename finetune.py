import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import bc_resnet_model
import apply
import custom_data


# ================================
# 설정
# ================================
PRETRAINED_MODEL = './example_model/model-sc-2.pt'
CHECKPOINT_FILE  = 'best_horiya'
SCALE            = 2      # 사전학습 모델과 동일하게!
N_CLASS          = 3      # silence, unknown, wakeword
FREEZE_BACKBONE  = True   # True: head만 학습 / False: 전체 학습
BATCH_SIZE       = 64
N_EPOCH          = 5
LR               = 0.001
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data   = data.to(device)
            target = target.to(device)
            output = model(data)
            pred   = output.squeeze().argmax(dim=-1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)
    return correct / total


if __name__ == "__main__":

    print(f"Device: {DEVICE}")

    # ================================
    # 1. 사전학습 모델 로드 (35개 클래스)
    # ================================
    print(f"\n사전학습 모델 로드: {PRETRAINED_MODEL}")
    model = bc_resnet_model.BcResNetModel(
        n_class=35,       # 원래 35개 클래스로 로드
        scale=SCALE,
        use_subspectral=True,
    ).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=DEVICE))
    print("사전학습 모델 로드 완료!")

    # ================================
    # 2. 분류기(head_conv)만 교체 (35 → 3)
    # ================================
    import torch.nn as nn
    model.head_conv = nn.Conv2d(32 * SCALE, N_CLASS, kernel_size=1).to(DEVICE)
    print(f"head_conv 교체 완료: 35 → {N_CLASS}")

    # ================================
    # 3. 백본 freeze (선택)
    # ================================
    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if 'head_conv' not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"백본 freeze: 학습 파라미터 {trainable:,} / 전체 {total:,}")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"전체 학습: 파라미터 {total:,}개")

    # ================================
    # 4. 데이터 로드
    # ================================
    print("\n데이터 로드 중...")
    train_files, val_files = custom_data.build_file_list()

    train_dataset = custom_data.CustomAudioDataset(train_files, is_training=True)
    val_dataset   = custom_data.CustomAudioDataset(val_files,   is_training=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=custom_data.collate_fn,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=custom_data.collate_fn,
        num_workers=2
    )

    # ================================
    # 5. 학습
    # ================================
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc   = 0.0
    best_model = copy.deepcopy(model)

    for epoch in range(N_EPOCH):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCH}")

        for data, target in loop:
            data   = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            loss   = F.nll_loss(output.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val_acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1} | Val Acc: {val_acc*100:.2f}%", end="")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), f"{CHECKPOINT_FILE}_{epoch+1}.pt")
            print(" (New Best!)")
        else:
            print()

    print(f"\n학습 완료! Best Val Acc: {best_acc*100:.2f}%")
    print(f"모델 저장: {CHECKPOINT_FILE}")
