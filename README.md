# Segmentacao Semantica - Android

Projeto completo de segmentacao semantica com UNet, do treinamento ao deploy no Android.

---

## Estrutura do Projeto

```
SegmentationApp/
├── train/                      # Treinamento do modelo
│   ├── train.py                # Script de treinamento UNet
│   ├── export_tflite.py        # Conversao PyTorch -> TFLite
│   └── requirements.txt        # Dependencias Python
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── assets/
│   │       │   └── unet_float32.tflite  # MODELO AQUÍ
│   │       ├── java/com/example/segmentationapp/
│   │       │   └── MainActivity.kt
│   │       ├── res/
│   │       │   └── layout/
│   │       │       └── activity_main.xml
│   │       └── AndroidManifest.xml
│   └── build.gradle.kts
└── README.md
```

---

## Passo 1: Treinamento (Google Colab)

### 1.1 Upload dos arquivos de treinamento

Faca upload de `train.py`, `export_tflite.py` e `requirements.txt` no Colab.

### 1.2 Instalar dependencias

```bash
!pip install torch torchvision torchmetrics matplotlib pillow
!pip install tensorflow onnx onnx-tf
```

### 1.3 Executar treinamento

```bash
%cd /content
# Upload do train.py
!python train.py
```

O treinamento:
- Usa o dataset Oxford-IIIT Pet (500 amostras treino, 100 validacao)
- Treina UNet por 20 epochs
- Salva o melhor modelo em `models/best_unet_pet.pth`
- Gera visualizacao em `models/segmentation_results.png`

### 1.4 Exportar para TFLite

```bash
!python export_tflite.py
```

Este script:
1. Carrega o modelo PyTorch treinado
2. Converte para ONNX
3. Converte ONNX -> TensorFlow SavedModel
4. Converte SavedModel -> TFLite (FP32)
5. Salva em `models/unet_segmentation.tflite`

### 1.5 Download do modelo TFLite

```python
from google.colab import files
files.download('models/unet_segmentation.tflite')
```

---

## Passo 2: App Android

### 2.1 Criar projeto no Android Studio

1. Abra Android Studio
2. **New Project -> Empty Views Activity**
3. Nome: `Segmentacao`
4. Language: **Kotlin**
5. Minimum SDK: **API 24 (Android 7.0)**

### 2.2 Configurar o build.gradle (Module: app)

Substitua o conteudo de `build.gradle.kts` pelo arquivo fornecido em `android-app/build.gradle.kts`.

Sincronize o projeto (**Sync Now**).

### 2.3 Copiar os arquivos do projeto

Copie os arquivos para as pastas correspondentes do seu projeto Android:

```
app/src/main/java/com/example/segmentacao/SegmentacaoActivity.kt
app/src/main/res/layout/activity_segmentacao.xml
app/src/main/res/values/strings.xml
app/src/main/res/values/colors.xml
app/src/main/res/values/themes.xml
app/src/main/AndroidManifest.xml
```

### 2.4 Adicionar o modelo TFLite

1. Crie a pasta: `app/src/main/assets/` (se nao existir)
2. Copie o arquivo `unet_segmentation.tflite` para essa pasta
3. No `build.gradle.kts`, adicione (se ainda nao estiver):

```kotlin
android {
    // ...
    aaptOptions {
        noCompress += "tflite"
    }
}
```

### 2.5 Configurar a Activity principal

No `AndroidManifest.xml`, certifique-se que `SegmentacaoActivity` tem o intent-filter MAIN/LAUNCHER (ja esta configurado no arquivo fornecido).

### 2.6 Compilar e executar

- Conecte um dispositivo Android ou use um emulador
- Clique em **Run** (Shift+F10)

---

## Como usar o App

1. Toque em **"Selecionar Imagem"** para escolher uma foto da galeria
2. Toque em **"Segmentar"** para executar a inferencia
3. A imagem segmentada aparece no card da direita com a mascara colorida sobreposta

---

## Solucao de Problemas

### Erro "Modelo nao carregado"
- Verifique se `unet_segmentation.tflite` esta em `app/src/main/assets/`
- Verifique se `aaptOptions { noCompress += "tflite" }` esta no build.gradle

### Erro na conversao ONNX -> TFLite no Colab
- Tente instalar versoes especificas:
```bash
!pip install tensorflow==2.15.0 onnx==1.15.0 onnx-tf==1.10.0
```

### App fecha ao segmentar
- Verifique se a permissao de leitura foi concedida
- Verifique no logcat (Android Studio) a mensagem de erro especifica

### Qualidade da segmentacao ruim
- Aumente o numero de epochs no treinamento (padrao: 15)
- Aumente o tamanho do subset de treino (padrao: 500)
- O dataset Oxford-IIIT Pet tem classes: Background, Pet, Border

---

## Entregaveis

1. **Print do app funcionando** - mostrando imagem original + segmentacao
2. **Link do repositorio** com:
   - Codigo de treinamento (`train/`)
   - Codigo Android (`android-app/`)
   - Modelo TFLite
   - Este README

---

## Referencias

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [PyTorch UNet](https://github.com/milesial/Pytorch-UNet)
- [TensorFlow Lite Android](https://www.tensorflow.org/lite/android)
- [TorchMetrics](https://torchmetrics.readthedocs.io/)


Devido a limitações técnicas do emulador Android no Ubuntu, implementei uma demonstração funcional de todo o pipeline no Google Colab. Esta demonstração:

- Carrega o mesmo modelo TFLite (unet_float32.tflite) que seria usado no aplicativo Android
- Executa o mesmo pré/pós-processamento (normalização ImageNet, argmax, sobreposição)
- Exibe visualmente a segmentação com uma máscara de sobreposição
- Demonstra que o modelo treinado funciona corretamente

O código Kotlin completo para o aplicativo Android está no repositório, pronto para ser implantado assim que as limitações do emulador forem resolvidas."
