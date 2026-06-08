# convert_local.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnx2tf # Import onnx2tf
import tensorflow as tf

# ========== DEFINIÇÃO COMPLETA DA UNET ==========
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
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        # Encoder (downsampling)
        self.inc = DoubleConv(3, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        self.d3 = Down(256, 512)
        self.d4 = Down(512, 512)

        # Decoder (upsampling)
        self.u1 = Up(1024, 256)   # 512+512 = 1024
        self.u2 = Up(512, 128)     # 256+256 = 512
        self.u3 = Up(256, 64)      # 128+128 = 256
        self.u4 = Up(128, 64)      # 64+64 = 128

        # Output layer
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 64
        x2 = self.d1(x1)      # 128
        x3 = self.d2(x2)      # 256
        x4 = self.d3(x3)      # 512
        x5 = self.d4(x4)      # 512

        # Decoder com skip connections
        x = self.u1(x5, x4)   # 256
        x = self.u2(x, x3)    # 128
        x = self.u3(x, x2)    # 64
        x = self.u4(x, x1)    # 64

        # Output
        return self.out(x)

# ========== CARREGAR E CONVERTER ==========
print("Carregando modelo...")
device = torch.device('cpu')
model = UNet(n_classes=3)

# Carregar os pesos do seu modelo treinado
# FIX: Load from the correct path 'models/model_final.pth'
model.load_state_dict(torch.load('models/model_final.pth', map_location=device))
model.eval()
print(" Modelo carregado com sucesso!")

# Testar modelo
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f" Teste: input shape {dummy_input.shape} \u2192 output shape {output.shape}")

# ========== EXPORTAR PARA ONNX ==========
print("\nExportando para ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "unet.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=18, # Updated opset_version
    do_constant_folding=True
)
print(" ONNX salvo: unet.onnx")

# ========== CONVERTER ONNX PARA TFLITE ==========
print("\nConvertendo ONNX para TFLite...")

try:
    # import onnx_tf # No longer needed directly

    # Carregar modelo ONNX
    onnx_model = onnx.load("unet.onnx")
    onnx.checker.check_model(onnx_model)
    print(" Modelo ONNX válido")

    # Converter ONNX para TensorFlow using onnx2tf
    output_tf_model_path = "tf_model"
    onnx2tf.convert(
        input_onnx_file_path="unet.onnx",
        output_folder_path=output_tf_model_path,
        non_verbose=True
    )
    print(" Modelo TensorFlow salvo")

    # Converter TensorFlow para TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(output_tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    # Input shapes for from_saved_model is often inferred, but can be set if needed.
    # For onnx2tf output, the input names usually match the ONNX graph.
    converter.input_shapes = {'input': [1, 224, 224, 3]} # Assuming 'input' is the input name and it expects NHWC

    tflite_model = converter.convert()

    # Salvar TFLite
    with open('unet_segmentation.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f" TFLite criado! Tamanho: {len(tflite_model) / 1024:.2f} KB")

except Exception as e:
    print(f" Erro na conversão automática: {e}")
    print("\n Método alternativo:")
    print("1. Use o arquivo 'unet.onnx' gerado")
    print("2. Acesse: https://coral.ai/models/convert/")
    print("3. Faça upload do unet.onnx")
    print("4. Converta para TFLite")
    print("5. Baixe o arquivo .tflite")

print("\n Script finalizado!")
