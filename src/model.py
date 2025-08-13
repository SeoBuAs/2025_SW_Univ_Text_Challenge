"""
Model classes and functions for AI text detection
"""
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel
)
from typing import Dict, Any, Tuple, Optional

from src.deberta_model import Custom_DebertaV2ForSequenceClassification
from src.electra_model import Custom_ElectraForSequenceClassification

def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    설정에 따라 모델과 토크나이저를 로드하는 함수
    이진 분류에 최적화: num_labels=1

    Args:
        config (Dict[str, Any]): 설정 딕셔너리

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: 모델과 토크나이저
    """
    model_names = config['model']['model_names']

    model = None
    tokenizer = None
    selected_model_name = None

    # 모델을 순서대로 시도
    for model_name in model_names:
        try:
            print(f"Attempting to load model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if model_name == "snunlp/KR-ELECTRA-discriminator" or model_name == "kykim/electra-kor-base":
                model = Custom_ElectraForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=1,
                    ignore_mismatched_sizes=True
                ).to('cuda' if torch.cuda.is_available() else 'cpu')
                selected_model_name = model_name
                print(f"Successfully loaded model: {model_name} (binary classification)")
                break
            else:
                model = Custom_DebertaV2ForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=1,
                    ignore_mismatched_sizes=True
                ).to('cuda' if torch.cuda.is_available() else 'cpu')
                selected_model_name = model_name
                print(f"Successfully loaded model: {model_name} (binary classification)")
                break
        except Exception as e:
            print(f"Failed to load model {model_name}: {str(e)}")
            continue

    if model is None or tokenizer is None:
        raise ValueError(f"Failed to load any of the specified models: {model_names}")

    print(f"Using model for binary classification: {selected_model_name}")
    return model, tokenizer


class AITextClassifier(nn.Module):
    """
    AI 생성 텍스트 탐지를 위한 커스텀 분류기
    이진 분류에 최적화: Linear(hidden_size, 1) + Sigmoid
    """

    def __init__(
        self,
        model_name: str,
        dropout_rate: float = 0.3,
        hidden_size: Optional[int] = None
    ):
        """
        Args:
            model_name (str): 사전 훈련된 모델 이름
            dropout_rate (float): 드롭아웃 비율
            hidden_size (Optional[int]): 히든 레이어 크기 (None이면 자동 설정)
        """
        super(AITextClassifier, self).__init__()

        # 백본 모델 로드
        self.backbone = AutoModel.from_pretrained(model_name)

        # 히든 사이즈 설정
        if hidden_size is None:
            hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 1)

        # 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids (torch.Tensor): 입력 토큰 ID
            attention_mask (torch.Tensor): 어텐션 마스크

        Returns:
            torch.Tensor: 이진 분류를 위한 로짓 (sigmoid 적용 전)
        """
        # 백본 모델을 통한 특성 추출
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS 토큰의 표현 또는 풀러 아웃풋 사용
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # CLS 토큰 (첫 번째 토큰) 사용
            pooled_output = outputs.last_hidden_state[:, 0, :]

        # 드롭아웃 적용
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits


def get_model_for_trainer(config: Dict[str, Any]) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Trainer를 사용한 훈련을 위한 모델과 토크나이저를 가져오는 함수
    (baseline.py 스타일)

    Args:
        config (Dict[str, Any]): 설정 딕셔너리

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: 모델과 토크나이저
    """
    return load_model_and_tokenizer(config)


def get_model_for_manual_training(config: Dict[str, Any]) -> Tuple[AITextClassifier, AutoTokenizer]:
    """
    수동 훈련을 위한 커스텀 모델과 토크나이저를 가져오는 함수
    (bert.py 스타일, 이진 분류 최적화)

    Args:
        config (Dict[str, Any]): 설정 딕셔너리

    Returns:
        Tuple[AITextClassifier, AutoTokenizer]: 커스텀 모델과 토크나이저
    """
    model_names = config['model']['model_names']
    dropout_rate = config['model']['dropout_rate']

    tokenizer = None
    selected_model_name = None

    # 토크나이저를 순서대로 시도
    for model_name in model_names:
        try:
            print(f"Attempting to load tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            selected_model_name = model_name
            print(f"Successfully loaded tokenizer: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load tokenizer {model_name}: {str(e)}")
            continue

    if tokenizer is None:
        raise ValueError(f"Failed to load any of the specified tokenizers: {model_names}")

    model = AITextClassifier(
        model_name=selected_model_name,
        dropout_rate=dropout_rate
    )

    print(f"Created binary classification model with backbone: {selected_model_name}")
    return model, tokenizer


def save_model(model, tokenizer, save_path: str, config: Dict[str, Any] = None) -> None:
    """
    모델과 토크나이저를 저장하는 함수

    Args:
        model: 저장할 모델
        tokenizer: 저장할 토크나이저
        save_path (str): 저장 경로
        config (Dict[str, Any], optional): 설정 딕셔너리
    """
    import os
    import json

    os.makedirs(save_path, exist_ok=True)

    # 모델 저장
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(save_path)
    else:
        torch.save(model.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))

    tokenizer.save_pretrained(save_path)

    if config:
        with open(os.path.join(save_path, 'training_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Model and tokenizer saved to: {save_path}")


def load_saved_model(
    load_path: str,
    model_class=None,
    device: torch.device = None
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    저장된 모델과 토크나이저를 로드하는 함수

    Args:
        load_path (str): 로드 경로
        model_class: 커스텀 모델 클래스 (필요한 경우)
        device (torch.device): 디바이스

    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: 로드된 모델과 토크나이저
    """
    import os

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    # 모델 로드
    if os.path.exists(os.path.join(load_path, 'pytorch_model.bin')):
        if model_class is None:
            raise ValueError("model_class must be provided for custom models")

        config_path = os.path.join(load_path, 'training_config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            model = model_class(
                model_name=config['model']['model_names'][0],
                dropout_rate=config['model']['dropout_rate']
            )
        else:
            model = model_class(
                model_name='klue/bert-base',
                dropout_rate=0.3
            )

        state_dict = torch.load(
            os.path.join(load_path, 'pytorch_model.bin'),
            map_location=device
        )
        model.load_state_dict(state_dict)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(load_path)

    if device:
        model.to(device)

    print(f"Model and tokenizer loaded from: {load_path}")
    return model, tokenizer


def load_model_and_tokenizer_for_eval(model_path: str):
    """
    저장된 모델과 토크나이저를 로드하는 함수

    Args:
        model_path (str): 모델 경로

    Returns:
        tuple: (model, tokenizer)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading model and tokenizer from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return model, tokenizer