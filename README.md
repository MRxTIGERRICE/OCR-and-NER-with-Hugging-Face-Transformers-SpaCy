# OCR and NER with Hugging Face Transformers & SpaCy

This project combines Optical Character Recognition (OCR) and Named Entity Recognition (NER) using a hybrid approach with Hugging Face Transformers and SpaCy. The aim is to extract meaningful entities from structured text data present in images, transforming them into machine-readable formats. The project covers the entire pipeline from training and evaluation to deployment of NER models on custom datasets.

## Features

- **Data Parsing**: Efficient parsing from TSV files containing image and text annotations.
- **Custom Training Pipelines**: Utilizes Hugging Face's BERT and SpaCy models to train for NER tasks.
- **Model Evaluation**: Implements precision, recall, and F1 score computations to assess model performance.
- **OCR Integration**: Seamlessly integrates OCR results into custom TSV datasets for downstream processing.
- **Data Conversion Utilities**: Includes utilities for converting data formats and saving model predictions.

## Setup

- **Dependencies**: Python, Pandas, SpaCy, Hugging Face Transformers, and other relevant libraries.
  
Install the required packages using:

```bash
pip install -r requirements.txt
