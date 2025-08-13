"""
Dataset classes for AI text detection
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Optional, Dict, Any
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os


def tokenize_single_row(args):
    """
    단일 행을 토크나이징하는 함수 (병렬 처리용)
    
    Args:
        args: (row_data, text_columns, target_column, is_test, tokenizer_name, max_length)
    
    Returns:
        Dict[str, torch.Tensor]: 토크나이즈된 데이터
    """
    row_data, text_columns, target_column, is_test, tokenizer_name, max_length = args
    
    # 각 프로세스에서 토크나이저 로드 (pickle 문제 해결)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 텍스트 결합 (title [SEP] full_text)
    texts = []
    for col in text_columns:
        if col in row_data and pd.notna(row_data[col]):
            texts.append(str(row_data[col]))
        else:
            texts.append('')
    
    combined_text = ' [SEP] '.join(texts)
    
    # 토크나이징
    encoding = tokenizer(
        combined_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    item = {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
    }
    
    if not is_test and target_column and target_column in row_data:
        item['labels'] = torch.tensor(row_data[target_column], dtype=torch.float)
    
    return item


class AITextDataset(Dataset):
    """
    AI 생성 텍스트 탐지를 위한 데이터셋 클래스
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        text_columns: List[str] = ["title", "full_text", "cluster"],
        target_column: Optional[str] = "generated",
        max_length: int = 512,
        is_test: bool = False,
        batch_size: int = 1000
    ):
        """
        Args:
            data_path (str): 데이터 파일 경로
            tokenizer (AutoTokenizer): 토크나이저
            text_columns (List[str]): 텍스트 컬럼 이름들
            target_column (Optional[str]): 타겟 컬럼 이름
            max_length (int): 최대 시퀀스 길이
            is_test (bool): 테스트 데이터 여부
            batch_size (int): 배치 토크나이징 크기
        """
        self.data = pd.read_csv(data_path, encoding='utf-8-sig')
        self.tokenizer = tokenizer
        self.text_columns = text_columns
        self.target_column = target_column
        self.max_length = max_length
        self.is_test = is_test
        self.batch_size = batch_size
        
        # 결측값 처리
        for col in text_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna('')
        
        # 테스트 데이터의 컬럼명 통일 (paragraph_text -> full_text)
        if 'paragraph_text' in self.data.columns and 'full_text' not in self.data.columns:
            self.data = self.data.rename(columns={'paragraph_text': 'full_text'})
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        if not is_test and target_column[0] and target_column[1] in self.data.columns:
            print(f"Target distribution: {self.data[target_column].value_counts().to_dict()}")
            print("Target Cluster distribution: ", self.data['cluster'].value_counts().to_dict() if 'cluster' in self.data.columns else "No cluster data")
        # 모든 데이터 미리 토크나이징 (배치 처리)
        print("Preprocessing and tokenizing all data...")
        self.encoded_data = self._preprocess_all_data_batch()
        print("Data preprocessing completed!")
    
    def _preprocess_all_data_batch(self) -> List[Dict[str, torch.Tensor]]:
        """
        모든 데이터를 배치 단위로 토크나이징하여 리스트로 저장
        
        Returns:
            List[Dict[str, torch.Tensor]]: 토크나이즈된 데이터 리스트
        """
        # 모든 텍스트를 미리 결합
        combined_texts = []
        labels = []
        cluster = []
        #for idx in range(100):
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            
            # 텍스트 결합 (title [SEP] full_text)
            texts = []
            for col in self.text_columns:
                if col in row and pd.notna(row[col]):
                    texts.append(str(row[col]))
                else:
                    texts.append('')
            
            combined_text = ' [SEP] '.join(texts)
            combined_texts.append(combined_text)
            
            # 레이블 수집
            if not self.is_test and self.target_column[0] and self.target_column[1] and self.target_column[0] and self.target_column[1] in row:
                labels.append(row[self.target_column[0]])
                cluster.append(row[self.target_column[1]])
        # 배치 단위로 토크나이징
        encoded_data = []
        num_batches = (len(combined_texts) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(num_batches), desc="Tokenizing batches"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(combined_texts))
            
            batch_texts = combined_texts[start_idx:end_idx]
            
            # 배치 토크나이징
            batch_encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 각 샘플별로 분리
            for j in range(len(batch_texts)):
                item = {
                    'input_ids': batch_encoding['input_ids'][j],
                    'attention_mask': batch_encoding['attention_mask'][j],
                }
                
                # 레이블 추가
                if not self.is_test and labels:
                    item['labels'] = torch.tensor(labels[start_idx + j], dtype=torch.float)
                    item['cluster'] = torch.tensor(cluster[start_idx + j], dtype=torch.long)

                encoded_data.append(item)
        
        return encoded_data
    
    def __len__(self) -> int:
        return len(self.encoded_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        미리 토크나이즈된 데이터에서 하나의 샘플을 가져오는 함수
        
        Args:
            idx (int): 인덱스
            
        Returns:
            Dict[str, torch.Tensor]: 토크나이즈된 입력과 레이블
        """
        return self.encoded_data[idx]


def create_dataloaders(
    config: Dict[str, Any],
    tokenizer: AutoTokenizer
) -> tuple:
    """
    데이터로더들을 생성하는 함수
    
    Args:
        config (Dict[str, Any]): 설정 딕셔너리
        tokenizer (AutoTokenizer): 토크나이저
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    
    # 훈련 데이터 로드
    train_dataset = AITextDataset(
        data_path=config['data']['train_file'],
        tokenizer=tokenizer,
        text_columns=config['data']['text_columns'],
        target_column=config['data']['target_column'],
        max_length=config['data']['max_length'],
        is_test=False
    )
    
    # 검증 데이터 로드
    val_dataset = AITextDataset(
        data_path=config['data']['val_file'],
        tokenizer=tokenizer,
        text_columns=config['data']['text_columns'],
        target_column=config['data']['target_column'],
        max_length=config['data']['max_length'],
        is_test=False
    )
    
    # 테스트 데이터 로드
    test_dataset = AITextDataset(
        data_path=config['data']['test_file'],
        tokenizer=tokenizer,
        text_columns=config['data']['text_columns'],
        target_column=None,
        max_length=config['data']['max_length'],
        is_test=True
    )
    
    # 동적 패딩을 위한 데이터 콜레이터
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['dataloader_num_workers'],
        pin_memory=torch.cuda.is_available(),
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['dataloader_num_workers'],
        pin_memory=torch.cuda.is_available(),
        collate_fn=data_collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['dataloader_num_workers'],
        pin_memory=torch.cuda.is_available(),
        collate_fn=data_collator
    )
    
    print(f"Created dataloaders:")
    print(f"  - Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  - Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  - Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def create_data_collator(tokenizer: AutoTokenizer):
    """
    동적 패딩을 위한 데이터 콜레이터 생성
    
    Args:
        tokenizer (AutoTokenizer): 토크나이저
        
    Returns:
        function: 데이터 콜레이터 함수
    """
    def collate_fn(batch):
        # 이미 토크나이즈된 데이터이므로 단순히 배치로 합치기
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # 레이블이 있는 경우
        if 'labels' in batch[0]:
            labels = torch.stack([item['labels'] for item in batch])
            result['labels'] = labels
        
        return result
    
    return collate_fn 