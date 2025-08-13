"""
Utility functions for AI text detection
"""

import os
import random
import logging
from typing import Dict, Any, Optional

import wandb
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from transformers import trainer_utils


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드하는 함수
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        Dict[str, Any]: 설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed: int = 42) -> None:
    """
    모든 랜덤 시드를 고정하는 함수
    
    Args:
        seed (int): 시드 값
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    로깅 설정을 초기화하는 함수
    
    Args:
        log_dir (str): 로그 디렉토리
        log_level (str): 로그 레벨
        
    Returns:
        logging.Logger: 설정된 로거
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 로거 생성
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 핸들러가 이미 있다면 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 파일 핸들러
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'training.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_device(device_preference: str = "auto") -> torch.device:
    """
    사용할 디바이스를 결정하는 함수
    
    Args:
        device_preference (str): 디바이스 선호도 ("auto", "cpu", "cuda", "mps")
        
    Returns:
        torch.device: 사용할 디바이스
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
    else:
        device = torch.device(device_preference)
        print(f"Using specified device: {device}")
    
    return device

CLASS_NAMES = ['Human', 'AI']


def compute_metrics(eval_pred: trainer_utils.EvalPrediction) -> dict:
    """
    평가 메트릭을 계산하고, 시각화는 wandb에 직접 로깅하는 함수
    """
    predictions, labels = eval_pred

    # 예측값 및 레이블 처리 (기존과 동일)
    if predictions.shape[-1] == 1:
        pred_probabilities = torch.sigmoid(torch.tensor(predictions)).squeeze().numpy()
        pred_labels = (pred_probabilities > 0.5).astype(int)
        labels = labels.astype(int)
    elif predictions.shape[-1] == 2:
        probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)
        pred_probabilities = probabilities[:, 1].numpy()
        pred_labels = np.argmax(predictions, axis=-1)
    else:
        pred_probabilities = predictions
        pred_labels = (predictions > 0.5).astype(int)

    auc = roc_auc_score(labels, pred_probabilities)
    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    mcc = matthews_corrcoef(labels, pred_labels)


    import time
    current_time = int(time.time())
    
    wandb.log({
        "eval/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=labels,
            preds=pred_labels,
            class_names=CLASS_NAMES
        ),
        "eval/confusion_matrix_timestamp": current_time,
        "eval/epoch_auc": auc,  # 추가 컨텍스트
        "eval/epoch_accuracy": accuracy,
        "eval/epoch_mcc": mcc
    })

    cm = confusion_matrix(labels, pred_labels)
    cm_text = f"Confusion Matrix:\n{cm}\nAccuracy: {accuracy:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}"
    
    wandb.log({
        "eval/confusion_matrix_text": wandb.Html(f"<pre>{cm_text}</pre>"),
    })
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
    }


def print_model_info(model, tokenizer) -> None:
    """
    모델 정보를 출력하는 함수
    
    Args:
        model: 모델 객체
        tokenizer: 토크나이저 객체
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Model name: {model.config.name_or_path}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Max sequence length: {tokenizer.model_max_length}")
    print("="*50 + "\n")


def save_predictions(predictions: np.ndarray, 
                    sample_submission_path: str, 
                    output_path: str) -> None:
    """
    예측 결과를 저장하는 함수
    
    Args:
        predictions (np.ndarray): 예측 확률
        sample_submission_path (str): 샘플 제출 파일 경로
        output_path (str): 출력 파일 경로
    """
    import pandas as pd
    
    sample_submission = pd.read_csv(sample_submission_path)
    

    if 'target' in sample_submission.columns:
        sample_submission['target'] = predictions
    else:
        sample_submission.iloc[:, 1] = predictions
    
    # 결과 저장
    sample_submission.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def create_directories(config: Dict[str, Any]) -> None:
    """
    필요한 디렉토리들을 생성하는 함수
    
    Args:
        config (Dict[str, Any]): 설정 딕셔너리
    """
    dirs_to_create = [
        config['output']['model_save_dir'],
        config['output']['results_dir'],
        config['output']['logs_dir']
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}") 