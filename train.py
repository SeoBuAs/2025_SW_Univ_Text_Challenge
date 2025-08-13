#!/usr/bin/env python3
"""
AI Text Detection Training Script (Trainer-based)
베이스라인 코드의 개선된 버전
"""

import os
import sys
import argparse
from datetime import datetime
import shutil

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback, Qwen3ForSequenceClassification
)
import torch.nn as nn

# 로컬 모듈 임포트
from src.utils import (
    load_config, seed_everything, setup_logging,
    get_device, compute_metrics, print_model_info,
    create_directories
)
from src.model import get_model_for_trainer, save_model
from src.dataset import create_dataloaders


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='AI Text Detection Training')
    parser.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA'
    )
    return parser.parse_args()


def setup_training_args(config: dict, output_dir: str) -> TrainingArguments:
    """훈련 인수 설정"""

    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(
            config['output']['model_save_dir'],
            f"ai_text_detection_{timestamp}"
        )

    return TrainingArguments(
        output_dir=output_dir,

        # 훈련 설정
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],

        # 평가 설정
        eval_strategy=config['evaluation']['strategy'],
        eval_steps=50 if config['evaluation']['strategy'] == "steps" else None,
        save_strategy=config['evaluation']['strategy'],
        save_steps=50 if config['evaluation']['strategy'] == "steps" else None,
        # 최적 모델 선택
        load_best_model_at_end=True,
        metric_for_best_model=config['evaluation']['metric'],
        # 로깅 설정
        logging_dir=config['output']['logs_dir'],
        logging_steps=10,
        logging_strategy="steps",
        # 하드웨어 최적화
        bf16=True,
        dataloader_num_workers=config['hardware']['dataloader_num_workers'],
        # 기타 설정
        remove_unused_columns=False,
        report_to=None,  # wandb 등 비활성화
        save_total_limit=10,  # 최대 10개 체크포인트만 유지
        seed=config['data']['random_state'],
        # Gradient clipping
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        # 메모리 최적화
        group_by_length=True,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=2,
    )


def main():
    """메인 훈련 함수"""

    # 명령행 인수 파싱
    args = parse_args()

    # 설정 로드
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    # 디렉토리 생성
    create_directories(config)

    # 시드 고정
    seed_everything(config['data']['random_state'])

    # 로깅 설정
    logger = setup_logging(config['output']['logs_dir'])
    logger.info("Starting AI Text Detection Training")

    # 디바이스 설정
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = get_device(config['hardware']['device'])

    # 모델과 토크나이저 로드
    logger.info("Loading model and tokenizer...")
    model, tokenizer = get_model_for_trainer(config)
    model.to(device)

    # 모델 정보 출력
    print_model_info(model, tokenizer)

    # 데이터로더 생성
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)

    # 데이터 콜레이터 설정
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 훈련 인수 설정
    training_args = setup_training_args(config, args.output_dir)
    logger.info(f"Training arguments configured. Output dir: {training_args.output_dir}")

    # 가중치가 적용된 커스텀 Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # 훈련 시작
    logger.info("Starting trainpiping...")
    print("\n" + "=" * 60)
    print("🚀 TRAINING STARTED")
    print("=" * 60)

    train_result = trainer.train()

    # 훈련 결과 출력
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)
    logger.info("Training completed successfully")

    # 최종 평가
    logger.info("Running final evaluation...")
    eval_result = trainer.evaluate()

    print(f"\n📊 FINAL RESULTS:")
    print(f"   - Best Validation AUC: {eval_result.get('eval_auc', 'N/A'):.4f}")
    print(f"   - Best Validation Accuracy: {eval_result.get('eval_accuracy', 'N/A'):.4f}")
    print(f"   - Training Runtime: {train_result.metrics.get('train_runtime', 'N/A'):.2f} seconds")

    # 모델 저장
    logger.info("Saving best model...")
    best_model_path = os.path.join(training_args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    # 설정 파일도 함께 저장
    shutil.copy(args.config, os.path.join(best_model_path, "config.yaml"))

    print(f"✅ Best model saved to: {best_model_path}")
    logger.info(f"Best model saved to: {best_model_path}")

    print("\n🎉 Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 