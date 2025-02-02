# Twitter_bot_detection
# Bot Detection System

## Overview

The Bot Detection System is a machine learning model designed to classify social media accounts as either "bots" or "humans" based on their tweets and behavioral features. This system leverages Natural Language Processing (NLP) and behavioral analysis to detect and differentiate between bot and human activity on platforms like Twitter.

## Features

- **Text Analysis**: Utilizes pre-trained BERT models (DistilBERT) to extract features from tweet content.
- **Behavioral Analysis**: Extracts features from user activity such as tweet frequency, engagement, follower count, and more.
- **Neural Network**: A custom neural network architecture combining both text and behavioral features to predict bot activity.

## Technologies

- **PyTorch**: Deep learning framework used for model training and inference.
- **Transformers (Hugging Face)**: Utilized for pre-trained BERT models to process text.
- **Pandas**: Data manipulation and feature extraction.
- **Scikit-learn**: Used for preprocessing, including feature scaling and splitting data.
- **Numpy**: Numerical operations.
- **CUDA (Optional)**: Enables GPU acceleration when available for faster training.

## Requirements

The following libraries are required to run this project:

- torch
- transformers
- pandas
- numpy
- scikit-learn
- tqdm
- joblib
- logging
- matplotlib (optional for visualizations)

You can install these dependencies using `pip`:

```bash
pip install torch transformers pandas numpy scikit-learn tqdm joblib matplotlib

MIT License

Copyright (c) 2025 [Utkarsh Verma]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
