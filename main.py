import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix
import logging
from pathlib import Path
import re
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
from torch.cuda.amp import autocast, GradScaler
import json
import joblib
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch
from collections import *
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
@dataclass
class ModelConfig:
    """Configuration for the bot detection model"""
    # Model architecture
    hidden_size = 256
    num_layers = 2
    dropout_rate = 0.3
    batch_size = 64
    learning_rate = 1e-4
    n_epochs = 20
    early_stopping_patience = 5
    max_text_length = 128
    bert_model = "distilbert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    use_fp16 = True

config = ModelConfig()
class TextProcessor:
    """Process text data for bot detection"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self.model = AutoModel.from_pretrained(config.bert_model)
        self.model.to(config.device)
        self.model.eval()
    
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """Extract BERT embeddings from text"""
        features = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting text features"):
                inputs = self.tokenizer(
                    text,
                    max_length=self.config.max_text_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(self.config.device)
                
                outputs = self.model(**inputs)
                # Use CLS token embedding
                features.append(outputs.last_hidden_state[:, 0, :].cpu())
        
        return torch.cat(features, dim=0)
class BehavioralFeatureExtractor:
    """Extract behavioral features from Twitter user activity"""
    
    @staticmethod
    def extract_features(data: pd.DataFrame) -> Dict[str, np.ndarray]:
        features = {}
        
        # Convert 'Created At' to datetime
        try:
            data['Created At'] = pd.to_datetime(data['Created At'], format='%d-%m-%Y %H:%M')
        except ValueError:
            # If the specific format fails, try automatic parsing
            data['Created At'] = pd.to_datetime(data['Created At'])
        
        # Time-based features
        max_date = data['Created At'].max()
        account_age = (max_date - data['Created At']).dt.total_seconds() / (24 * 3600)
        
        # Post frequency features
        features['post_frequency'] = 1 / np.maximum(1, account_age)  # Posts per day
        
        # Engagement metrics
        features['retweet_engagement'] = data['Retweet Count'] / np.maximum(1, data['Follower Count'])
        features['mention_rate'] = data['Mention Count'] / np.maximum(1, account_age)
        features['follower_impact'] = np.log1p(data['Follower Count'])  # Log-transformed follower count
        
        # Verification status
        features['is_verified'] = data['Verified'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0)
        
        # Content complexity
        features['tweet_length'] = data['Tweet'].str.len()
        features['words_per_tweet'] = data['Tweet'].str.split().str.len()
        
        # Hashtag features
        features['hashtag_density'] = data['Hashtags'].str.count('#').fillna(0)
        
        # Location features
        features['has_location'] = (data['Location'].notna() & (data['Location'] != '')).astype(int)
        
        # Time pattern features
        data['hour'] = data['Created At'].dt.hour
        features['posting_hours_variance'] = data.groupby('User ID')['hour'].transform('std').fillna(0)
        
        # Calculate posting intervals
        data = data.sort_values(['User ID', 'Created At'])
        data['prev_post_time'] = data.groupby('User ID')['Created At'].shift(1)
        data['post_interval'] = (data['Created At'] - data['prev_post_time']).dt.total_seconds() / 3600
        features['std_posting_interval'] = data.groupby('User ID')['post_interval'].transform('std').fillna(0)
        
        return {k: np.nan_to_num(np.array(v), 0) for k, v in features.items()}
class BotDetectionDataset(Dataset):
    """PyTorch dataset for bot detection"""
    def __init__(self, text_features: torch.Tensor, behavioral_features: torch.Tensor, labels: torch.Tensor):
        self.text_features = text_features
        self.behavioral_features = behavioral_features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'text_features': self.text_features[idx],
            'behavioral_features': self.behavioral_features[idx],
            'label': self.labels[idx]
        }
class BotDetectionModel(nn.Module):
    """Neural network for bot detection"""
    def __init__(self, config: ModelConfig, text_dim: int, behavioral_dim: int):
        super().__init__()
        
        # Text processing branch
        self.text_network = nn.Sequential(
            nn.Linear(text_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Behavioral processing branch
        self.behavioral_network = nn.Sequential(
            nn.Linear(behavioral_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Combined processing
        self.combined_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, behavioral_features):
        text_out = self.text_network(text_features)
        behavioral_out = self.behavioral_network(behavioral_features)
        combined = torch.cat([text_out, behavioral_out], dim=1)
        return self.combined_network(combined)
class BotDetector:
    """Main class for bot detection"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_processor = TextProcessor(config)
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        # Initialize AMP scaler in __init__ for better organization
        self.amp_scaler = GradScaler() if self.config.use_fp16 and torch.cuda.is_available() else None
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training"""
        # Extract features
        logger.info("Extracting text features...")
        # Changed from data['text'] to data['Tweet']
        text_features = self.text_processor.extract_text_features(data['Tweet'].tolist())
        
        logger.info("Extracting behavioral features...")
        behavioral_features = pd.DataFrame(self.behavioral_extractor.extract_features(data))
        
        # Scale behavioral features
        behavioral_features = self.scaler.fit_transform(behavioral_features)
        behavioral_features = torch.FloatTensor(behavioral_features)
        
        # Use 'is_bot' as it was created in main()
        labels = torch.FloatTensor(data['is_bot'].values)
        
        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=0.2,
            random_state=42
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            random_state=42
        )
        
        # Create datasets
        train_dataset = BotDetectionDataset(
            text_features[train_idx],
            behavioral_features[train_idx],
            labels[train_idx]
        )
        val_dataset = BotDetectionDataset(
            text_features[val_idx],
            behavioral_features[val_idx],
            labels[val_idx]
        )
        test_dataset = BotDetectionDataset(
            text_features[test_idx],
            behavioral_features[test_idx],
            labels[test_idx]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train the model"""
        # Get dimensions from first batch
        text_dim = next(iter(train_loader))['text_features'].shape[1]
        behavioral_dim = next(iter(train_loader))['behavioral_features'].shape[1]
        
        # Initialize model
        self.model = BotDetectionModel(
            self.config,
            text_dim,
            behavioral_dim
        ).to(self.device)  # Use self.device instead of config.device
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        best_val_f1 = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(self.config.n_epochs):
            # Training
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.n_epochs}'):
                text_features = batch['text_features'].to(self.device)
                behavioral_features = batch['behavioral_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Fixed autocast implementation with proper device_type
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with autocast(device_type=device_type, enabled=self.config.use_fp16):
                    outputs = self.model(text_features, behavioral_features)
                    loss = criterion(outputs, labels.unsqueeze(1))
                
                if self.amp_scaler is not None:
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.step(optimizer)
                    self.amp_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Update history
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_metrics['f1'])
            
            logger.info(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, Val F1 = {val_metrics['f1']:.4f}"
            )
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model"""
        self.model.eval()
        all_losses = []
        all_preds = []
        all_labels = []
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for batch in dataloader:
                text_features = batch['text_features'].to(self.config.device)
                behavioral_features = batch['behavioral_features'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                outputs = self.model(text_features, behavioral_features)
                loss = criterion(outputs, labels.unsqueeze(1))
                
                all_losses.append(loss.item())
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, np.round(all_preds)),
            'precision': precision_score(all_labels, np.round(all_preds)),
            'recall': recall_score(all_labels, np.round(all_preds)),
            'f1': f1_score(all_labels, np.round(all_preds)),
            'auc_roc': roc_auc_score(all_labels, all_preds)
        }
        
        return np.mean(all_losses), metrics
    
    def predict(self, text: str, behavioral_data: Dict[str, float]) -> Dict[str, float]:
        """Make prediction for a single instance"""
        self.model.eval()
        
        with torch.no_grad():
            # Process text
            text_features = self.text_processor.extract_text_features([text])
            
            # Process behavioral features
            behavioral_features = pd.DataFrame([behavioral_data])
            behavioral_features = self.scaler.transform(behavioral_features)
            behavioral_features = torch.FloatTensor(behavioral_features)
            
            # Make prediction
            text_features = text_features.to(self.config.device)
            behavioral_features = behavioral_features.to(self.config.device)
            output = self.model(text_features, behavioral_features)
            
            confidence = float(output.cpu().numpy()[0][0])
            
        return {
            'is_bot': confidence >= 0.5,
            'confidence': confidence
        }
    
    def save_model(self, path: str):
        """Save model and preprocessing components"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_state': joblib.dump(self.scaler,'scaler.joblib'),
            'config': self.config
        }, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'BotDetector':
        """Load a saved model"""
        checkpoint = torch.load(path)
        detector = cls(checkpoint['config'])
        detector.scaler = joblib.loads(checkpoint['scaler_state'])
        detector.model = BotDetectionModel(
            detector.config,
            text_dim=768,  # BERT base hidden size
            behavioral_dim=len(detector.behavioral_extractor.extract_features(pd.DataFrame()))
        ).to(detector.config.device)
        detector.model.load_state_dict(checkpoint['model_state_dict'])
        return detector
def main():
    """Main function to train and evaluate the bot detection system"""
    try:
        # Initialize configuration
        config = ModelConfig()
        
        # Initialize detector
        detector = BotDetector(config)
        
        # Load data
        logger.info("Loading and preprocessing data...")
        data = pd.read_csv("output.csv")
        
        # Verify required columns
        required_columns = [
            'User ID', 'Username', 'Tweet', 'Retweet Count', 'Mention Count',
            'Follower Count', 'Verified', 'Bot Label', 'Location', 'Created At', 'Hashtags'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Basic data cleaning
        data['Tweet'] = data['Tweet'].fillna('')
        data['Location'] = data['Location'].fillna('')
        data['Hashtags'] = data['Hashtags'].fillna('')
        data['Verified'] = data['Verified'].fillna('FALSE')
        data['Retweet Count'] = data['Retweet Count'].fillna(0)
        data['Mention Count'] = data['Mention Count'].fillna(0)
        data['Follower Count'] = data['Follower Count'].fillna(0)
        
        # Convert label column
        data['is_bot'] = data['Bot Label'].astype(float)
        
        # Prepare data loaders
        logger.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = detector.prepare_data(data)
        
        # Train the model
        logger.info("Training started")
        history = detector.train(train_loader, val_loader)
        
        # Evaluate on the test set  
        logger.info("Evaluating on test set...")
        test_loss, test_metrics = detector.evaluate(test_loader)
        logger.info(f"Test Metrics: {test_metrics}")
        
        # Save the model
        detector.save_model("bot_detection_model.pth")
        logger.info("Model saved successfully")

        
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
if __name__ == "__main__":
    main()