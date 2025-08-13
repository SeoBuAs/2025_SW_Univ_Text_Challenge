#!/usr/bin/env python3
"""
Qwen3-Embedding-0.6B 기반 문서 클러스터링 시스템
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import argparse
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
import pickle

# 필요한 라이브러리 임포트
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import hdbscan
    import umap.umap_ as umap
    UMAP = umap.UMAP
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"필요한 패키지를 설치해주세요: {e}")
    print("pip install transformers>=4.51.0 torch scikit-learn hdbscan umap-learn plotly")
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

class DocumentClusteringSystem:
    """Qwen3-Embedding-0.6B 기반 문서 클러스터링 시스템"""
    
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
        self.max_length = 8192
        self.documents = []
        self.embeddings = None
        self.clusters = None
        self.cluster_labels = None
        self.reduced_embeddings = None
        self.results = {}
        self.original_df = None  # 원본 데이터프레임 저장용
        self.valid_indices = []  # 유효한 문서 인덱스 저장용
        
        # 주제 분류를 위한 task description (영어)
        self.task_description = "Analyze the topic and content of documents to classify them by subject and group similar content and topics together"
        
        print(f"🚀 Document Clustering System 초기화")
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
            
    def load_and_preprocess_documents(self, csv_path: str, text_column: str = 'full_text', 
                                    sample_size: Optional[int] = None) -> List[str]:
        """
        문서 수집 (title + full_text 결합, 전처리 없음)
        
        Args:
            csv_path: CSV 파일 경로
            text_column: 텍스트가 들어있는 컬럼명 (사용하지 않음, title+full_text 결합)
            sample_size: 샘플링할 문서 수 (None이면 전체 사용)
            
        Returns:
            문서 리스트 (title + full_text 결합, 전처리 없음)
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
            return []
        
        # 샘플링 (필요시)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"🎯 샘플링 완료: {len(df)}개 문서")
        
        # 원본 데이터프레임 저장 (클러스터링 결과 추가용)
        self.original_df = df.copy()
        
        # 텍스트 결합 (Qwen3-Embedding 최적화)
        print("🔧 title + full_text 결합 중... (주제 클러스터링 최적화)")
        documents = []
        
        for idx, row in tqdm(df.iterrows(), desc="문서 처리", total=len(df)):
            title = str(row['title']) if pd.notna(row['title']) else ""
            full_text = str(row['full_text']) if pd.notna(row['full_text']) else ""
            
            # title과 full_text를 간단하게 결합 (Qwen3-Embedding이 자동으로 최적화)
            if title and full_text:
                combined_text = f"This is the title: {title}\n\nThis is the full text: {full_text}"
            elif title:
                combined_text = title
            elif full_text:
                combined_text = full_text
            else:
                combined_text = ""
            
            documents.append(combined_text)
        
        self.documents = documents
        valid_count = len([d for d in documents if d.strip()])
        print(f"✅ 문서 처리 완료: {valid_count}개 유효 문서 / {len(documents)}개 전체")
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정제 (한국어 특화)"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL 제거
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 이메일 제거
        text = re.sub(r'\S+@\S+', '', text)
        
        # 한국어, 영어, 숫자, 공백만 유지 (특수문자 제거)
        text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]', ' ', text)
        
        # 연속된 공백 통일
        text = re.sub(r'\s+', ' ', text)
        
        # 줄바꿈 정리
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # 너무 짧은 단어 제거 (1글자 단어)
        words = text.split()
        words = [word for word in words if len(word) > 1]
        text = ' '.join(words)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def generate_embeddings(self, batch_size: int = 4) -> Optional[np.ndarray]:
        """
        임베딩 생성 (Qwen3-Embedding 직접 사용)
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            정규화된 임베딩 배열 또는 None (실패시)
        """
        if not self.documents:
            print("❌ 문서가 없습니다. 먼저 문서를 로드해주세요.")
            return None
            
        if self.tokenizer is None or self.model is None:
            self.load_embedding_model()
            
        # 모델 로딩 실패 체크
        if self.tokenizer is None or self.model is None:
            print("❌ 임베딩 모델 로딩에 실패했습니다.")
            return None
        
        print(f"🔄 임베딩 생성 중... (배치 크기: {batch_size})")
        print(f"🚀 고사양 시스템 최적화 모드 (512GB RAM, 48GB VRAM)")
        
        try:
            # 유효한 문서만 임베딩 생성
            valid_documents = [doc for doc in self.documents if doc.strip()]
            valid_indices = [i for i, doc in enumerate(self.documents) if doc.strip()]
            
            print(f"📊 유효 문서 수: {len(valid_documents)} / {len(self.documents)}")
            
            if not valid_documents:
                print("❌ 유효한 문서가 없습니다.")
                return None
            
            # 주제 분류를 위한 쿼리 텍스트 생성
            query_texts = []
            for doc in valid_documents:
                query_text = get_detailed_instruct(self.task_description, doc)
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
                valid_embeddings = np.vstack(all_embeddings)
            else:
                print("❌ 임베딩 생성 실패")
                return None
            
            # 전체 문서 수에 맞게 임베딩 배열 생성 (빈 문서는 0으로 채움)
            embedding_dim = valid_embeddings.shape[1]
            all_embeddings = np.zeros((len(self.documents), embedding_dim), dtype=np.float32)
            
            # 유효한 문서의 임베딩을 올바른 위치에 배치
            for i, valid_idx in enumerate(valid_indices):
                all_embeddings[valid_idx] = valid_embeddings[i]
            
            self.embeddings = all_embeddings
            self.valid_indices = valid_indices  # 유효한 인덱스 저장
            
            print(f"✅ 임베딩 생성 완료: {all_embeddings.shape}")
            print(f"📊 유효 임베딩: {len(valid_indices)}개")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return all_embeddings
        except Exception as e:
            print(f"❌ 임베딩 생성 실패: {e}")
            return None
    
    def load_embeddings_from_files(self, embeddings_dir: str, file_pattern: str = "*.csv") -> Optional[np.ndarray]:
        """
        저장된 임베딩 파일들에서 임베딩 로드
        
        Args:
            embeddings_dir: 임베딩 파일들이 저장된 디렉토리
            file_pattern: 파일 패턴 (예: "*.csv", "*.pkl")
            
        Returns:
            임베딩 배열 또는 None (실패시)
        """
        import glob
        import ast
        
        print(f"📥 임베딩 파일 로딩: {embeddings_dir}")
        
        # 파일 목록 가져오기
        file_path = os.path.join(embeddings_dir, file_pattern)
        embedding_files = sorted(glob.glob(file_path))
        
        if not embedding_files:
            print(f"❌ 임베딩 파일을 찾을 수 없습니다: {file_path}")
            return None
        
        print(f"📁 발견된 임베딩 파일 수: {len(embedding_files)}")
        
        all_embeddings = []
        all_documents = []
        all_data = []
        
        # 각 파일에서 임베딩 로드
        for file_path in tqdm(embedding_files, desc="임베딩 파일 로딩"):
            try:
                if file_path.endswith('.csv'):
                    # CSV 파일 로드
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    
                    # 임베딩 컬럼 확인
                    if 'qwen3_embedding' not in df.columns:
                        print(f"⚠️ 임베딩 컬럼이 없습니다: {file_path}")
                        continue
                    
                    # 임베딩 문자열을 리스트로 변환
                    embeddings = []
                    for emb_str in df['qwen3_embedding']:
                        try:
                            if isinstance(emb_str, str):
                                # 문자열을 리스트로 파싱
                                emb_list = ast.literal_eval(emb_str)
                                embeddings.append(np.array(emb_list, dtype=np.float32))
                            else:
                                embeddings.append(np.array(emb_str, dtype=np.float32))
                        except:
                            # 파싱 실패시 0으로 채움
                            embeddings.append(np.zeros(1024, dtype=np.float32))
                    
                    embeddings = np.array(embeddings)
                    
                elif file_path.endswith('.pkl'):
                    # Pickle 파일 로드
                    with open(file_path, 'rb') as f:
                        df = pickle.load(f)
                    
                    # 임베딩 추출
                    embeddings = np.array([np.array(emb, dtype=np.float32) for emb in df['qwen3_embedding']])
                
                else:
                    print(f"⚠️ 지원하지 않는 파일 형식: {file_path}")
                    continue
                
                # 문서 정보 수집
                if 'title' in df.columns and 'full_text' in df.columns:
                    for _, row in df.iterrows():
                        title = str(row['title']) if pd.notna(row['title']) else ""
                        full_text = str(row['full_text']) if pd.notna(row['full_text']) else ""
                        
                        if title and full_text:
                            combined_text = f"This is the title: {title}\n\nThis is the full text: {full_text}"
                        elif title:
                            combined_text = title
                        elif full_text:
                            combined_text = full_text
                        else:
                            combined_text = ""
                        
                        all_documents.append(combined_text)
                
                # 데이터 수집
                all_data.append(df)
                all_embeddings.append(embeddings)
                
                print(f"✅ {file_path}: {len(embeddings)}개 임베딩 로드")
                
            except Exception as e:
                print(f"❌ 파일 로드 실패 {file_path}: {e}")
                continue
        
        if not all_embeddings:
            print("❌ 로드된 임베딩이 없습니다.")
            return None
        
        # 모든 임베딩 결합
        self.embeddings = np.vstack(all_embeddings)
        self.documents = all_documents
        
        # 원본 데이터프레임 결합
        if all_data:
            self.original_df = pd.concat(all_data, ignore_index=True)
        
        # 유효한 인덱스 설정 (모든 임베딩이 유효하다고 가정)
        self.valid_indices = list(range(len(self.embeddings)))
        
        print(f"✅ 임베딩 로드 완료:")
        print(f"   - 총 임베딩 수: {len(self.embeddings)}")
        print(f"   - 임베딩 차원: {self.embeddings.shape[1]}")
        print(f"   - 문서 수: {len(self.documents)}")
        print(f"   - 원본 데이터: {len(self.original_df) if self.original_df is not None else 0}개 행")
        
        return self.embeddings
    
    def perform_clustering(self, algorithm: str = 'hdbscan', **kwargs) -> Optional[np.ndarray]:
        """
        클러스터링 수행 (유효한 문서만 클러스터링)
        
        Args:
            algorithm: 클러스터링 알고리즘 ('kmeans', 'hdbscan')
            **kwargs: 알고리즘별 파라미터
            
        Returns:
            클러스터 라벨 배열 또는 None (실패시)
        """
        if self.embeddings is None:
            print("❌ 임베딩이 없습니다. 먼저 임베딩을 생성해주세요.")
            return None
            
        if not hasattr(self, 'valid_indices'):
            print("❌ 유효한 인덱스 정보가 없습니다.")
            return None
            
        print(f"🎯 클러스터링 수행: {algorithm.upper()}")
        
        try:
            # 유효한 문서의 임베딩만 추출
            valid_embeddings = self.embeddings[self.valid_indices]
            
            if algorithm.lower() == 'kmeans':
                n_clusters = kwargs.get('n_clusters', 8)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                valid_cluster_labels = clusterer.fit_predict(valid_embeddings)
                
            elif algorithm.lower() == 'hdbscan':
                min_cluster_size = kwargs.get('min_cluster_size', 15)
                min_samples = kwargs.get('min_samples', 5)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                valid_cluster_labels = clusterer.fit_predict(valid_embeddings)
                
            else:
                print(f"❌ 지원하지 않는 알고리즘: {algorithm}")
                return None
            
            # 전체 문서 수에 맞게 클러스터 라벨 배열 생성 (무효한 문서는 -1로 설정)
            all_cluster_labels = np.full(len(self.documents), -1, dtype=int)
            
            # 유효한 문서의 클러스터 라벨을 올바른 위치에 배치
            for i, valid_idx in enumerate(self.valid_indices):
                all_cluster_labels[valid_idx] = valid_cluster_labels[i]
            
            self.cluster_labels = all_cluster_labels
            
            # 클러스터링 결과 분석
            n_clusters = len(set(valid_cluster_labels)) - (1 if -1 in valid_cluster_labels else 0)
            n_noise = list(valid_cluster_labels).count(-1)
            n_invalid = len(self.documents) - len(self.valid_indices)
            
            print(f"📊 클러스터링 결과:")
            print(f"   - 클러스터 수: {n_clusters}")
            print(f"   - 노이즈 포인트: {n_noise}")
            print(f"   - 무효 문서: {n_invalid}")
            print(f"   - 유효 문서: {len(self.valid_indices)}")
            
            # 실루엣 스코어 계산 (노이즈 포인트 제외)
            if n_clusters > 1:
                mask = valid_cluster_labels != -1
                if np.sum(mask) > 1:
                    silhouette_avg = silhouette_score(valid_embeddings[mask], valid_cluster_labels[mask])
                    print(f"   - 실루엣 스코어: {silhouette_avg:.3f}")
            
            return all_cluster_labels
            
        except Exception as e:
            print(f"❌ 클러스터링 실패: {e}")
            return None
    
    def analyze_clusters(self) -> Dict:
        """클러스터 결과 분석"""
        if self.cluster_labels is None:
            print("❌ 클러스터링이 수행되지 않았습니다.")
            return {}
        
        print("📈 클러스터 분석 중...")
        
        analysis = {}
        unique_labels = set(self.cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 노이즈 클러스터
                continue
                
            cluster_docs = [self.documents[i] for i, l in enumerate(self.cluster_labels) if l == label]
            
            analysis[label] = {
                'size': len(cluster_docs),
                'documents': cluster_docs[:5],  # 샘플 문서 5개
                'avg_length': np.mean([len(doc) for doc in cluster_docs])
            }
        
        self.results['cluster_analysis'] = analysis
        
        # 클러스터별 크기 출력
        print(f"📊 클러스터별 문서 수:")
        for label, info in analysis.items():
            print(f"   클러스터 {label}: {info['size']}개 문서")
        
        return analysis
    
    def reduce_dimensions(self, n_components: int = 3) -> Optional[np.ndarray]:
        """차원 축소 (3차원 시각화용)"""
        if self.embeddings is None:
            print("❌ 임베딩이 없습니다.")
            return None
            
        print(f"🔄 차원 축소 중... ({n_components}차원)")
        
        try:
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            
            reduced_embeddings = reducer.fit_transform(self.embeddings)
            
            # Convert to numpy array and ensure proper dtype
            reduced_embeddings = np.asarray(reduced_embeddings, dtype=np.float32)
            
            self.reduced_embeddings = reduced_embeddings
            
            print(f"✅ 차원 축소 완료: {reduced_embeddings.shape}")
            return reduced_embeddings
        except Exception as e:
            print(f"❌ 차원 축소 실패: {e}")
            return None
    
    def visualize_clusters(self, save_path: Optional[str] = None):
        """클러스터링 결과 시각화 (3차원, 투명 배경)"""
        if self.reduced_embeddings is None:
            print("🔄 차원 축소가 필요합니다...")
            if self.reduce_dimensions(n_components=3) is None:
                print("❌ 차원 축소 실패로 시각화를 중단합니다.")
                return
        
        if self.cluster_labels is None:
            print("❌ 클러스터링이 수행되지 않았습니다.")
            return
        
        if self.reduced_embeddings is None:
            print("❌ 차원 축소된 임베딩이 없습니다.")
            return
        
        print("🎨 클러스터링 결과 시각화 중... (3차원, 투명 배경)")
        
        try:
            # 3차원 시각화를 위한 설정
            fig = plt.figure(figsize=(15, 10))
            
            # 투명 배경 설정
            fig.patch.set_facecolor('none')
            
            # 3D 서브플롯 생성
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('none')
            
            # 클러스터별 색상 지정
            unique_labels = set(self.cluster_labels)
            colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:  # 노이즈 포인트
                    color = 'black'
                    marker = 'x'
                    alpha = 0.5
                    s = 30
                else:
                    marker = 'o'
                    alpha = 0.7
                    s = 50
                
                mask = self.cluster_labels == label
                ax.scatter(
                    self.reduced_embeddings[mask, 0],
                    self.reduced_embeddings[mask, 1],
                    self.reduced_embeddings[mask, 2],
                    c=[color],
                    marker=marker,
                    alpha=alpha,
                    s=s,
                    label=f'Cluster {label}' if label != -1 else 'Noise'
                )
            
            ax.set_title('Document Clustering Results (3D UMAP)', fontsize=16, fontweight='bold')
            ax.set_xlabel('UMAP Component 1', fontsize=12, fontweight='bold')
            ax.set_ylabel('UMAP Component 2', fontsize=12, fontweight='bold')
            ax.set_zlabel('UMAP Component 3', fontsize=12, fontweight='bold')
            
            # 범례 설정
            ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
            
            # 격자 설정
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='none', edgecolor='none', transparent=True)
                print(f"💾 시각화 저장: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 시각화 실패: {e}")
            if save_path:
                print(f"💡 시각화 파일 저장 실패: {save_path}")
    
    def save_results(self, output_dir: str = "./clustering_results"):
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 클러스터링 결과 저장
        if self.cluster_labels is not None:
            results_df = pd.DataFrame({
                'document': self.documents,
                'cluster': self.cluster_labels
            })
            
            csv_path = output_path / f"clustering_results_{timestamp}.csv"
            results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"💾 클러스터링 결과 저장: {csv_path}")
        
        # 시각화 저장
        if self.reduced_embeddings is not None:
            viz_path = output_path / f"cluster_visualization_{timestamp}.png"
            self.visualize_clusters(save_path=str(viz_path))

    def save_clustering_results(self, output_path: str = "./train_clustering.csv"):
        """
        클러스터링 결과를 원본 데이터에 추가하여 저장
        
        Args:
            output_path: 저장할 파일 경로
        """
        if self.original_df is None or self.cluster_labels is None:
            print("❌ 원본 데이터 또는 클러스터링 결과가 없습니다.")
            return
        
        print("💾 클러스터링 결과 저장 중...")
        
        # 원본 데이터프레임 복사
        result_df = self.original_df.copy()
        
        # 클러스터링 결과 컬럼 추가
        result_df['cluster'] = self.cluster_labels
        
        # 클러스터링 통계 정보 출력
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        print(f"📊 클러스터별 문서 수:")
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:
                print(f"   노이즈/무효: {count}개")
            else:
                print(f"   클러스터 {cluster_id}: {count}개")
        
        # CSV 파일로 저장
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 클러스터링 결과 저장 완료: {output_path}")
        print(f"📋 저장된 컬럼: {list(result_df.columns)}")
        
        return result_df

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Qwen3-Embedding 기반 문서 클러스터링 (고사양 시스템 최적화)')
    parser.add_argument('--data', default='../data/train.csv', help='데이터 파일 경로')
    parser.add_argument('--embeddings-dir', default='./embeddings_output', help='임베딩 파일들이 저장된 디렉토리')
    parser.add_argument('--text-column', default='full_text', help='텍스트 컬럼명')
    parser.add_argument('--sample-size', type=int, help='샘플링할 문서 수')
    parser.add_argument('--algorithm', default='kmeans', choices=['kmeans'], 
                       help='클러스터링 알고리즘 (kmeans로 고정)')
    parser.add_argument('--n-clusters', type=int, default=5, help='클러스터 수 (5로 고정)')
    parser.add_argument('--min-cluster-size', type=int, default=10, help='최소 클러스터 크기 (HDBSCAN용)')
    parser.add_argument('--batch-size', type=int, default=4, help='임베딩 배치 크기 (고사양 시스템용)')
    parser.add_argument('--output-dir', default='./clustering_results', help='결과 저장 디렉토리')
    parser.add_argument('--clustering-output', default='./clustering_results/train_clustering.csv', help='클러스터링 결과 CSV 파일 경로')
    parser.add_argument('--no-visualization', action='store_true', help='시각화 건너뛰기')
    parser.add_argument('--use-saved-embeddings', action='store_true', default=True, help='저장된 임베딩 사용')
    
    args = parser.parse_args()
    
    # kmeans로 고정하고 k=5로 설정
    args.algorithm = 'kmeans'
    args.n_clusters = 5
    
    print("🚀 고사양 시스템 최적화 문서 클러스터링 시작!")
    print(f"💾 시스템 사양: 512GB RAM, 48GB VRAM")
    print(f"🎯 클러스터링 설정: K-means, k=5")
    
    # 클러스터링 시스템 초기화
    clustering_system = DocumentClusteringSystem()
    
    if args.use_saved_embeddings:
        # 저장된 임베딩 파일들에서 로드
        embeddings = clustering_system.load_embeddings_from_files(args.embeddings_dir)
        if embeddings is None:
            print("❌ 저장된 임베딩 로드 실패")
            return
    else:
        # 기존 방식: 문서 로드 → 임베딩 생성
        documents = clustering_system.load_and_preprocess_documents(
            args.data, 
            args.text_column, 
            args.sample_size
        )
        
        if not documents:
            print("❌ 처리할 문서가 없습니다.")
            return
        
        # 임베딩 생성 (고사양 시스템 최적화)
        embeddings = clustering_system.generate_embeddings(batch_size=args.batch_size)
    
    # 클러스터링 수행 (kmeans, k=5로 고정)
    cluster_labels = clustering_system.perform_clustering('kmeans', n_clusters=5)
    
    # 결과 분석
    clustering_system.analyze_clusters()
    
    # 시각화
    if not args.no_visualization:
        clustering_system.visualize_clusters()
    
    # 결과 저장
    clustering_system.save_results(args.output_dir)
    
    # 클러스터링 결과를 원본 데이터에 추가하여 저장
    clustering_system.save_clustering_results(args.clustering_output)
    
    print("🎉 문서 클러스터링 완료!")
    print(f"💡 고사양 시스템 활용으로 최적화된 성능 제공")

if __name__ == "__main__":
    main() 