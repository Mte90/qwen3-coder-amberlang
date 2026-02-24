# Guide to Fine-Tuning Qwen3-Coder for the Amber Programming Language

This guide will help you fine-tune the **Qwen3-Coder** model (a Qwen-based language model optimized for code generation) to support the [Amber programming language](https://amber-lang.com/).

The goal is to train the model with hundreds of Amber scripting examples, enabling it to:

- Generate Amber code
- Convert Bash or Python scripts to Amber

## Table of Contents

1. [Important Note](#important-note)
2. [Hardware Requirements](#hardware-requirements)
3. [Prerequisites](#prerequisites)
4. [Download Qwen3-Coder Model](#step-4-download-qwen3-coder-model)
5. [Prepare the Dataset](#step-5-prepare-the-dataset)
6. [Fine-Tuning Options](#fine-tuning-options)
7. [Test the Fine-Tuned Model](#step-7-test-the-fine-tuned-model)

## Important Note

This guide assumes basic knowledge of **Linux**, **Python**, and **machine learning**.

Fine-tuning a large model requires significant resources. To reduce memory requirements, this guide uses **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA**.

## Hardware Requirements

- **GPU**: NVIDIA with minimum **8GB VRAM** (tested on RTX 3060 with 12GB)
- **RAM**: Minimum **16GB** (31GB recommended for larger datasets)
- **Disk**: ~40GB free space for model download and checkpoints
- **CUDA**: Compatible with CUDA 11.8+

## Prerequisites

- Python 3.10+
- NVIDIA GPU with compatible drivers

Install dependencies by running `install.sh`:

```
chmod +x install.sh
./install.sh
```

## Step 4: Download Qwen3-Coder Model

The `huggingface-cli` tool downloads files directly to disk without loading them into memory, avoiding OOM issues on machines with limited RAM/VRAM.

Install `huggingface-cli` first (required for downloading without loading into memory):

```
source ./amber-finetune-env/bin/activate
pip install huggingface_hub[cli]
```

If you encounter authentication errors:

```
huggingface-cli login
```

## Step 5: Prepare the Dataset

Run:

```
source ./amber-finetune-env/bin/activate
./prepare_dataset.py /Amber/src/tests/stdlib/ /Amber/src/tests/translating/
```

## Fine-Tuning Options

Both workflows use the same dataset preparation. Choose one based on your hardware:

### Option A: Full Fine-Tuning (Recommended for 24GB+ VRAM)

Run training on the full model:

```
python train.py
```

This uses the existing `train.py` script which automatically loads the model in 4-bit if needed.

### Option B: LoRA Fine-Tuning (Recommended for 8-12GB VRAM)

The LoRA approach trains only ~1% of parameters, making it suitable for lower-VRAM GPUs.

1. **Update download script** for 8B (edit `download-qwen-coder.py`):
   - Use model name: `"Qwen/Qwen2.5-Coder-8B-Instruct"`

2. **Download the model**:

   ```
   python download-qwen-coder.py
   ```

   The model downloads to `./qwen3-coder-base`.

3. **Train with LoRA**:

   ```
   python train_lora_8b.py --epochs 3 --batch-size 1 --lr 5e-5
   ```

4. **Test**:

   ```
   python test.py
   ```

## Step 7: Test the Fine-Tuned Model

Run:

```
python test.py
```
