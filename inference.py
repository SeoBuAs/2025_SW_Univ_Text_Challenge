#!/usr/bin/env python3
"""
AI Text Detection Prediction Script
"""

import os
import sys
import argparse
from typing import List

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.deberta_model import Custom_DebertaV2ForSequenceClassification
from src.electra_model import Custom_ElectraForSequenceClassification
# 로컬 모듈 임포트
from src.utils import (
    load_config, seed_everything, get_device,
    save_predictions, setup_logging
)
from src.dataset import AITextDataset
from src.model import load_saved_model, AITextClassifier


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='AI Text Detection Prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
        help='Test data file path (overrides config)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output prediction file path'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for prediction (overrides config)'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA'
    )
    return parser.parse_args()


def predict_with_trainer_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    test_dataset: AITextDataset,
    device: torch.device,
    batch_size: int = 16
) -> np.ndarray:
    """
    Hugging Face 모델을 사용한 예측

    Args:
        model: 훈련된 모델
        tokenizer: 토크나이저
        test_dataset: 테스트 데이터셋
        device: 디바이스
        batch_size: 배치 크기

    Returns:
        np.ndarray: 예측 확률 (클래스 1의 확률)
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    model.eval()
    model.to(device)

    # 데이터로더 생성
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if outputs.logits.shape[-1] == 1:  # num_labels=1인 경우
                pred_probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
            else:
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_probs = probabilities[:, 1].cpu().numpy()

            predictions.extend(pred_probs)

    return np.array(predictions)


def predict_with_custom_model(
    model: AITextClassifier,
    tokenizer: AutoTokenizer,
    test_dataset: AITextDataset,
    device: torch.device,
    batch_size: int = 16
) -> np.ndarray:
    """
    커스텀 모델을 사용한 예측

    Args:
        model: 훈련된 커스텀 모델
        tokenizer: 토크나이저
        test_dataset: 테스트 데이터셋
        device: 디바이스
        batch_size: 배치 크기

    Returns:
        np.ndarray: 예측 확률 (클래스 1의 확률)
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    model.eval()
    model.to(device)

    # 데이터로더 생성
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            pred_probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            predictions.extend(pred_probs)

    return np.array(predictions)


def main():
    """메인 예측 함수"""

    args = parse_args()

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    seed_everything(config['data']['random_state'])

    logger = setup_logging(config['output']['logs_dir'])
    logger.info("Starting AI Text Detection Prediction")

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = get_device(config['hardware']['device'])

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    print(f"Loading model from: {args.model_path}")

    
    if args.model_path == './models/deberta/checkpoint-8232':
        model = Custom_DebertaV2ForSequenceClassification.from_pretrained(args.model_path)
    else:
        model = Custom_ElectraForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_type = "huggingface"

    # 테스트 데이터 경로 설정
    test_file = args.test_file if args.test_file else config['data']['test_file']

    # 테스트 데이터셋 생성
    print(f"Loading test data from: {test_file}")
    test_dataset = AITextDataset(
        data_path=test_file,
        tokenizer=tokenizer,
        text_columns=config['data']['text_columns'],
        target_column=None,
        max_length=config['data']['max_length'],
        is_test=True
    )

    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']

    logger.info("Starting prediction...")
    print("\n" + "="*50)
    print("🔮 PREDICTION STARTED")
    print("="*50)

    if model_type == "huggingface":
        predictions = predict_with_trainer_model(
            model, tokenizer, test_dataset, device, batch_size
        )

    print(f"\nGenerated {len(predictions)} predictions")
    print(f"Prediction statistics:")
    print(f"  - Mean: {predictions.mean():.4f}")
    print(f"  - Std: {predictions.std():.4f}")
    print(f"  - Min: {predictions.min():.4f}")
    print(f"  - Max: {predictions.max():.4f}")

    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            config['output']['results_dir'],
            f"predictions_{timestamp}.csv"
        )

    logger.info(f"Saving predictions to: {output_file}")

    sample_submission_path = config['data']['submission_file']

    try:
        save_predictions(predictions, sample_submission_path, output_file)
    except Exception as e:
        print(f"Warning: Could not use sample submission file ({e})")
        print("Creating prediction file directly...")

        result_df = pd.DataFrame({
            'ID': [f'TEST_{i:04d}' for i in range(len(predictions))],
            'generated': predictions
        })
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

    print("\n" + "="*50)
    print("✅ PREDICTION COMPLETED")
    print("="*50)
    logger.info("Prediction completed successfully")

    print(f"✅ Predictions saved to: {output_file}")
    print("\n🎉 Prediction pipeline completed successfully!")


if __name__ == "__main__":
    main() 