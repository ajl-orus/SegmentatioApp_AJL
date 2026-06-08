# complete_pipeline.py
# Script completo: treino + exportação

import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchmetrics.classification import MulticlassJaccardIndex, Accuracy
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIG =================
os.makedirs('./models', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ================= UNET MODEL =================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diffX//2, diffX-diffX//2,
                                   diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        self.d3 = Down(256, 512)
        self.d4 = Down(512, 512)

        self.u1 = Up(1024, 256)
        self.u2 = Up(512, 128)
        self.u3 = Up(256, 64)
        self.u4 = Up(128, 64)
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)

        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out(x)

# ================= TRANSFORMS =================
img_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def mask_tf(mask):
    mask = transforms.Resize((224,224),
            interpolation=transforms.InterpolationMode.NEAREST)(mask)
    mask = np.array(mask).astype(np.int64) - 1
    mask[mask < 0] = 0
    mask[mask > 2] = 2
    return torch.tensor(mask, dtype=torch.long)

# ================= DATASET =================
print("Carregando dataset...")
dataset = OxfordIIITPet(
    root='./data',
    split='trainval',
    target_types='segmentation',
    transform=img_tf,
    target_transform=mask_tf,
    download=True
)

# Usar mais dados para melhor treino
train_idx = torch.randperm(len(dataset))[:800]  # Aumentado para 800
train_ds = Subset(dataset, train_idx)

val_dataset = OxfordIIITPet(
    root='./data',
    split='test',
    target_types='segmentation',
    transform=img_tf,
    target_transform=mask_tf,
    download=True
)

val_ds = Subset(val_dataset, torch.randperm(len(val_dataset))[:150])  # Mais validação

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

# ================= TRAIN =================
print("\nIniciando treinamento...")
model = UNet(n_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

iou_metric = MulticlassJaccardIndex(num_classes=3).to(device)
acc_metric = Accuracy(task="multiclass", num_classes=3).to(device)

best_iou = 0

for epoch in range(15):  # Aumentado para 15 épocas
    model.train()
    iou_metric.reset()
    acc_metric.reset()
    loss_sum = 0

    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        pred = out.argmax(1)
        iou_metric.update(pred,y)
        acc_metric.update(pred,y)

    train_iou = iou_metric.compute().item()
    train_acc = acc_metric.compute().item()
    train_loss = loss_sum / len(train_loader)

    # Validação
    model.eval()
    val_iou_metric = MulticlassJaccardIndex(num_classes=3).to(device)

    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            val_iou_metric.update(pred,y)

    val_iou = val_iou_metric.compute().item()

    print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.3f} | Train IoU: {train_iou:.3f} | Val IoU: {val_iou:.3f}")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), 'models/best.pth')
        print(f"  -> Melhor modelo salvo! (IoU: {best_iou:.3f})")

    scheduler.step(val_iou)

print(f"\n Treinamento concluído! Melhor IoU: {best_iou:.3f}")

# ================= CONVERT TO TFLITE =================
print("\n" + "="*50)
print("Convertendo para TensorFlow Lite...")
print("="*50)

try:
    import tensorflow as tf
    import onnx
    import onnx2tf

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "models/unet_pet.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True
    )

    # Convert ONNX to TensorFlow
    onnx2tf.convert(
        input_onnx_file_path="models/unet_pet.onnx",
        output_folder_path="models/tf_model",
        non_verbose=True
    )

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("models/tf_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.input_shapes = {'input': [1, 224, 224, 3]}

    tflite_model = converter.convert()

    with open('models/unet_segmentation.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f" Modelo TFLite criado! Tamanho: {len(tflite_model)/1024:.2f} KB")

    # Download do arquivo
    from google.colab import files
    files.download('models/unet_segmentation.tflite')

except Exception as e:
    print(f" Conversão automática falhou: {e}")
    print("\n Você pode baixar o modelo .pth e converter manualmente")

    # Salvar apenas o modelo PyTorch
    torch.save(model.state_dict(), 'models/model_final.pth')
    from google.colab import files
    files.download('models/model_final.pth')

# ================= VISUALIZATION =================
print("\n" + "="*50)
print("Visualizando resultados...")
print("="*50)

model.eval()
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i in range(4):
    x,y = val_ds[i]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).argmax(1).cpu()[0]

    # Desnormalizar para visualização
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = x.permute(1,2,0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)

    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Imagem {i+1}")
    axes[0, i].axis('off')

    axes[1, i].imshow(pred, cmap='tab10', vmin=0, vmax=2)
    axes[1, i].set_title(f"Segmentação")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('models/resultados_segmentacao.png', dpi=150)
plt.show()

print("\n Pipeline completo finalizado!")
print(" Arquivos salvos em './models/'")
