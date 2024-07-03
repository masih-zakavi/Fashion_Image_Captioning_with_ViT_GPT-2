## Fashion Image Captioning with ViT and GPT-2

This repository contains the implementation of a specialized image captioning model for clothing items utilizing Vision Transformer (ViT) as the encoder and GPT-2 as the decoder. The project includes pretraining on the Flickr8k dataset and fine-tuning on the FashionGen dataset. More details on the structure of the model can be found in the overview section of the ViT_GPT_2.ipynb notebook.

### Notebooks

1. **ViT_GPT_2.ipynb**
   - Jupyter Notebook for pretraining the image captioning model on the Flickr8k dataset.
   - Includes data loading, model architecture setup, training loops, and initial evaluation.
   - A PDF version of this notebook is available as `ViT_GPT_2.pdf`.

2. **Fine_Tuning.ipynb**
   - Jupyter Notebook for fine-tuning the pretrained model on the FashionGen dataset.
   - Covers data processing, model fine-tuning, and final evaluation.
   - A PDF version of this notebook is available as `Fine_Tuning.pdf`.

### Scripts

- **Preprocess.py**
  - Python script for preprocessing the FashionGen dataset.
  - Utilizes Pandas and Pillow to unzip data from the .h5 file and clean the data for fine-tuning.

## Detailed Description

The Fashion Image Captioning model generates images specifically for fashion images. The model leverages the strengths of Vision Transformer (ViT) for visual feature extraction and GPT-2 for generating natural language descriptions. Key components and steps of the project include:

- **Data Preprocessing**: 
  - FashionGen dataset is preprocessed using `Preprocess.py`.
  - Data augmentation and cleaning steps are applied to improve performance.

- **Model Architecture**:
  - The model is designed with ViT as the encoder to capture visual details.
  - GPT-2 is used as the decoder to generate descriptions for the items.

- **Pretraining**:
  - Initial training is conducted on the Flickr8k dataset using `ViT_GPT_2.ipynb` to establish a robust base model.
  - This step includes setting up the training loop, loss functions, and evaluation metrics.

- **Fine-Tuning**:
  - The pretrained model is further fine-tuned on the FashionGen dataset in `Fine_Tuning.ipynb`.
  - Transfer Learning is employed to adapt the model to the specialized dataset.

- **Training and Optimization**:
  - Training is conducted on Google Cloud to leverage high computational power.
  - Customized dataset classes and data collators are used to optimize memory management during training.
