# Guide to Fine-Tuning Qwen3-Coder for the Amber Programming Language

This guide will help you fine-tune the **Qwen3-Coder** model (a Qwen-based language model optimized for code generation) to support the [Amber programming language](https://amber-lang.com/).

The goal is to train the model with hundreds of Amber scripting examples, enabling it to:

- Generate Amber code
- Convert Bash or Python scripts to Amber

## Important Note

This guide assumes basic knowledge of **Linux**, **Python**, and **machine learning**.

Fine-tuning a large model requires significant resources:

- CPU / RAM
- NVIDIA GPU (minimum **8GB VRAM**, recommended **24GB+**)

To reduce memory requirements, this guide uses **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA**.

## Step 2: Install NVIDIA Drivers and CUDA

No explanation needed

## Step 3: Create Python Virtual Environment and Install Libraries

Run `install.sh`.

## Step 4: Download Qwen3-Coder Model

The previous script does it!

If authentication is required:

```
huggingface-cli login
```

## Step 5: Prepare the Dataset

Run:

```
source ./amber-finetune-env/bin/activate
./prepare_dataset.py /Amber/src/tests/stdlib/ /Amber/src/tests/translating/
```

## Step 6: Perform Fine-Tuning

Run training:

```
python train.py
```

Note: Training may take hours or days depending on dataset and GPU. Monitor with nvidia-smi or htop.

## Step 7: Test the Fine-Tuned Model

Run:

```
python test.py
```
