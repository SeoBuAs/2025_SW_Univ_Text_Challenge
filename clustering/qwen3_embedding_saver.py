#!/usr/bin/env python3
"""
Qwen3-Embedding 임베딩 생성 및 배치 저장 시스템

이 스크립트는:
1. train.csv 데이터를 로드
2. Qwen3-Embedding으로 임베딩 생성 (transformers 직접 사용)
3. 메모리 오버플로우 방지를 위해 배치 단위로 저장
4. 원본 데이터 + 임베딩값을 pickle/csv로 저장
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 필요한 라이브러리 임포트
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    print(f"필요한 패키지를 설치해주세요: {e}")
    print("pip install transformers>=4.51.0 torch")
    exit(1)

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Last token pooling for Qwen3-Embedding"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Get detailed instruction for Qwen3-Embedding"""
    return f'Instruct: {task_description}\nQuery: {query}'

class EmbeddingSaver:
    """Qwen3-Embedding 임베딩 생성 및 배치 저장 시스템"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        """
        시스템 초기화
        
        Args:
            model_name: 사용할 임베딩 모델 이름
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_length = 4096
        self.data_df = None
        
        # 주제 분류를 위한 task description (영어)
        self.task_description = "Please analyze the topic and content of each document based on its title and content to enable future grouping of similar documents by subject."
        
        print(f"🚀 Embedding Saver 초기화")
        print(f"📊 사용 모델: {model_name}")
        print(f"🔧 디바이스: {self.device}")
        
    def load_embedding_model(self):
        """임베딩 모델 로드 (Qwen3-Embedding 직접 사용)"""
        try:
            print(f"📥 임베딩 모델 로딩 중: {self.model_name}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                padding_side='left'
            )
            
            # 모델 로드 (기본 설정)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            print(f"✅ 모델 로딩 완료")
                
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            self.tokenizer = None
            self.model = None
            
    def load_data(self, csv_path: str, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        데이터 로드
        
        Args:
            csv_path: CSV 파일 경로
            sample_size: 샘플링할 데이터 수 (None이면 전체 사용)
            
        Returns:
            로드된 데이터프레임
        """
        print(f"📂 데이터 로딩: {csv_path}")
        
        # 데이터 로드
        df = pd.read_csv(csv_path)
        print(f"📊 전체 데이터 수: {len(df)}")
        
        # 필수 컬럼 확인
        required_columns = ['title', 'full_text', 'generated']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 필수 컬럼을 찾을 수 없습니다: {missing_columns}")
            print(f"💡 사용 가능한 컬럼: {list(df.columns)}")
            return None
        
        # 샘플링 (필요시)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"🎯 샘플링 완료: {len(df)}개 데이터")
        
        self.data_df = df
        print(f"✅ 데이터 로드 완료: {len(df)}개 행")
        
        return df
    
    def prepare_text(self, title: str, full_text: str) -> str:
        """
        텍스트 준비 (title + full_text 결합)
        
        Args:
            title: 제목
            full_text: 본문
            
        Returns:
            결합된 텍스트
        """
        title = str(title) if pd.notna(title) else ""
        full_text = str(full_text) if pd.notna(full_text) else ""
        
        # title과 full_text를 간단하게 결합
        if title and full_text:
            combined_text = f"This is the title: {title}\n\nThis is the full text: {full_text}"
        elif title:
            combined_text = title
        elif full_text:
            combined_text = full_text
        else:
            combined_text = ""
        
        return combined_text
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        배치 단위로 임베딩 생성 (Qwen3-Embedding 직접 사용)
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            임베딩 배열
        """
        if self.tokenizer is None or self.model is None:
            self.load_embedding_model()
        
        if self.tokenizer is None or self.model is None:
            print("❌ 임베딩 모델 로딩 실패")
            return np.array([])
        
        print(f"🔄 임베딩 생성 중... (배치 크기: {batch_size})")
        
        # 주제 분류를 위한 쿼리 텍스트 생성
        query_texts = []
        for text in texts:
            query_text = get_detailed_instruct(self.task_description, text)
            query_texts.append(query_text)
        
        all_embeddings = []
        
        # 배치 단위로 처리
        for i in tqdm(range(0, len(query_texts), batch_size), desc="임베딩 생성"):
            batch_texts = query_texts[i:i + batch_size]
            
            try:
                # 토크나이징
                batch_dict = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                
                # 모델 추론
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    
                    # 임베딩 정규화
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    # CPU로 이동하여 저장
                    embeddings = embeddings.cpu().numpy()
                    all_embeddings.append(embeddings)
                    
            except Exception as e:
                print(f"❌ 배치 {i//batch_size + 1} 임베딩 생성 실패: {e}")
                # 빈 임베딩으로 대체
                empty_embeddings = np.zeros((len(batch_texts), 1024), dtype=np.float32)
                all_embeddings.append(empty_embeddings)
        
        # 모든 배치 결합
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
        else:
            final_embeddings = np.array([])
        
        print(f"✅ 임베딩 생성 완료: {final_embeddings.shape}")
        
        return final_embeddings
    
    def save_batch_data(self, batch_df: pd.DataFrame, batch_embeddings: np.ndarray, 
                       output_dir: str, batch_idx: int, save_format: str = 'pickle'):
        """
        배치 데이터 저장
        
        Args:
            batch_df: 배치 데이터프레임
            batch_embeddings: 배치 임베딩
            output_dir: 출력 디렉토리
            batch_idx: 배치 인덱스
            save_format: 저장 형식 ('pickle' 또는 'csv')
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 배치 데이터프레임에 임베딩 추가
        result_df = batch_df.copy()
        result_df['qwen3_embedding'] = [emb.tolist() for emb in batch_embeddings]
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_format.lower() == 'pickle':
            # Pickle 형식으로 저장
            filename = f"embeddings_batch_{batch_idx:04d}_{timestamp}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(result_df, f)
            
            print(f"💾 배치 {batch_idx} 저장 완료 (Pickle): {filepath}")
            
        elif save_format.lower() == 'csv':
            # CSV 형식으로 저장 (임베딩은 문자열로 변환)
            filename = f"embeddings_batch_{batch_idx:04d}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # 임베딩을 문자열로 변환하여 CSV 저장
            result_df['qwen3_embedding'] = result_df['qwen3_embedding'].apply(str)
            result_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            print(f"💾 배치 {batch_idx} 저장 완료 (CSV): {filepath}")
        
        # 메모리 정리
        del result_df
        
        return filepath
    
    def process_and_save(self, csv_path: str, save_interval: int = 10000, 
                        batch_size: int = 4, output_dir: str = "./embeddings_output",
                        save_format: str = 'pickle', sample_size: Optional[int] = None):
        """
        전체 프로세스 실행: 데이터 로드 → 임베딩 생성 → 배치 저장
        
        Args:
            csv_path: 입력 CSV 파일 경로
            save_interval: 저장 간격 (몇 개 데이터마다 저장할지)
            batch_size: 임베딩 생성 배치 크기
            output_dir: 출력 디렉토리
            save_format: 저장 형식 ('pickle' 또는 'csv')
            sample_size: 샘플링할 데이터 수
        """
        print("="*80)
        print("🚀 Qwen3-Embedding 배치 저장 시스템 시작")
        print("="*80)
        
        # 1. 데이터 로드
        df = self.load_data(csv_path, sample_size)
        if df is None:
            print("❌ 데이터 로드 실패")
            return []
        
        total_rows = len(df)
        num_batches = (total_rows + save_interval - 1) // save_interval
        
        print(f"📊 처리 계획:")
        print(f"   - 전체 데이터: {total_rows:,}개")
        print(f"   - 저장 간격: {save_interval:,}개")
        print(f"   - 예상 배치 수: {num_batches}개")
        print(f"   - 임베딩 배치 크기: {batch_size}")
        print(f"   - 저장 형식: {save_format.upper()}")
        print(f"   - 주제 분류 태스크: {self.task_description}")
        
        saved_files = []
        
        # 2. 배치별 처리
        for batch_idx in range(num_batches):
            start_idx = batch_idx * save_interval
            end_idx = min((batch_idx + 1) * save_interval, total_rows)
            
            print(f"\n🔄 배치 {batch_idx + 1}/{num_batches} 처리 중 ({start_idx:,}-{end_idx-1:,})")
            
            # 배치 데이터 추출
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            # 텍스트 준비
            print("📝 텍스트 준비 중...")
            texts = []
            for _, row in batch_df.iterrows():
                combined_text = self.prepare_text(row['title'], row['full_text'])
                texts.append(combined_text)
            
            # 임베딩 생성
            batch_embeddings = self.generate_embeddings_batch(texts, batch_size)
            
            if len(batch_embeddings) == 0:
                print(f"❌ 배치 {batch_idx + 1} 임베딩 생성 실패")
                continue
            
            # 배치 저장
            filepath = self.save_batch_data(
                batch_df, batch_embeddings, output_dir, 
                batch_idx + 1, save_format
            )
            saved_files.append(filepath)
            
            # 메모리 정리
            del batch_df, texts, batch_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\n" + "="*80)
        print("🎉 모든 배치 처리 완료!")
        print("="*80)
        print(f"📁 저장된 파일 수: {len(saved_files)}개")
        print(f"📂 출력 디렉토리: {output_dir}")
        print(f"💾 저장 형식: {save_format.upper()}")
        
        return saved_files

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Qwen3-Embedding 배치 저장 시스템')
    parser.add_argument('--data', default='/home/work/INC_share/GU/SWContest/trial_taemin/data/train.csv', help='입력 CSV 파일 경로')
    parser.add_argument('--save-interval', type=int, default=1000, help='저장 간격 (데이터 개수)')
    parser.add_argument('--batch-size', type=int, default=8, help='임베딩 생성 배치 크기')
    parser.add_argument('--output-dir', default='./embeddings_output', help='출력 디렉토리')
    parser.add_argument('--save-format', default='csv', choices=['pickle', 'csv'], 
                       help='저장 형식')
    parser.add_argument('--sample-size', type=int, help='샘플링할 데이터 수')
    parser.add_argument('--model-name', default='Qwen/Qwen3-Embedding-0.6B', 
                       help='사용할 임베딩 모델')
    
    args = parser.parse_args()
    
    # 임베딩 저장 시스템 초기화
    saver = EmbeddingSaver(model_name=args.model_name)
    
    # 전체 프로세스 실행
    saved_files = saver.process_and_save(
        csv_path=args.data,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_format=args.save_format,
        sample_size=args.sample_size
    )
    
    if saved_files:
        print(f"\n✅ 임베딩 저장 완료! 총 {len(saved_files)}개 파일 생성")
    else:
        print(f"\n❌ 임베딩 저장 실패")

if __name__ == "__main__":
    main() 