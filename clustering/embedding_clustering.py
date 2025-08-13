#!/usr/bin/env python3
"""
사전 생성된 임베딩을 이용한 클러스터링 시스템

이 스크립트는:
1. 사전 생성된 임베딩 CSV/Pickle 파일들을 로드
2. 임베딩을 결합하여 전체 데이터셋 구성
3. 클러스터링 알고리즘 적용 (K-means, HDBSCAN)
4. 시각화 및 결과 저장
"""

import pandas as pd
import numpy as np
import pickle
import os
import glob
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from tqdm import tqdm
import warnings
import ast
warnings.filterwarnings('ignore')

# 필요한 라이브러리 임포트
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import hdbscan
    import umap.umap_ as umap
    UMAP = umap.UMAP
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"필요한 패키지를 설치해주세요: {e}")
    print("pip install scikit-learn hdbscan umap-learn matplotlib seaborn plotly")
    exit(1)

class EmbeddingClusteringSystem:
    """사전 생성된 임베딩을 이용한 클러스터링 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.embeddings_df = None
        self.embeddings = None
        self.cluster_labels = None
        self.reduced_embeddings = None
        self.results = {}
        
        print(f"🚀 Embedding Clustering System 초기화")
        
    def load_embedding_files(self, input_dir: str, file_format: str = 'csv') -> Optional[pd.DataFrame]:
        """
        임베딩 파일들을 로드하여 하나의 데이터프레임으로 결합
        
        Args:
            input_dir: 임베딩 파일들이 있는 디렉토리
            file_format: 파일 형식 ('csv' 또는 'pickle')
            
        Returns:
            결합된 데이터프레임 또는 None (실패시)
        """
        print(f"📂 임베딩 파일 로딩 중: {input_dir}")
        print(f"📄 파일 형식: {file_format.upper()}")
        
        # 파일 패턴 설정
        if file_format.lower() == 'csv':
            file_pattern = os.path.join(input_dir, "embeddings_batch_*.csv")
        elif file_format.lower() == 'pickle':
            file_pattern = os.path.join(input_dir, "embeddings_batch_*.pkl")
        else:
            print(f"❌ 지원하지 않는 파일 형식: {file_format}")
            return None
        
        # 파일 목록 가져오기
        embedding_files = sorted(glob.glob(file_pattern))
        
        if not embedding_files:
            print(f"❌ 임베딩 파일을 찾을 수 없습니다: {file_pattern}")
            return None
        
        print(f"📁 발견된 파일 수: {len(embedding_files)}개")
        
        all_dataframes = []
        
        # 각 파일 로드
        for i, file_path in enumerate(tqdm(embedding_files, desc="파일 로딩")):
            try:
                if file_format.lower() == 'csv':
                    # CSV 파일 로드
                    df = pd.read_csv(file_path)
                    
                    # 임베딩 컬럼이 문자열로 저장되어 있으므로 리스트로 변환
                    if 'qwen3_embedding' in df.columns:
                        df['qwen3_embedding'] = df['qwen3_embedding'].apply(ast.literal_eval)
                    
                elif file_format.lower() == 'pickle':
                    # Pickle 파일 로드
                    with open(file_path, 'rb') as f:
                        df = pickle.load(f)
                
                all_dataframes.append(df)
                print(f"   ✅ 파일 {i+1}/{len(embedding_files)}: {len(df)}개 행 로드")
                
            except Exception as e:
                print(f"   ❌ 파일 {i+1} 로딩 실패: {e}")
                continue
        
        if not all_dataframes:
            print("❌ 로드된 파일이 없습니다.")
            return None
        
        # 모든 데이터프레임 결합
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"✅ 임베딩 로딩 완료")
        print(f"📊 전체 데이터 수: {len(combined_df):,}개")
        print(f"📋 컬럼: {list(combined_df.columns)}")
        
        # 임베딩 유효성 검사
        if 'qwen3_embedding' not in combined_df.columns:
            print("❌ 'qwen3_embedding' 컬럼을 찾을 수 없습니다.")
            return None
        
        # 임베딩 차원 확인
        first_embedding = combined_df['qwen3_embedding'].iloc[0]
        embedding_dim = len(first_embedding) if isinstance(first_embedding, list) else 0
        print(f"🔢 임베딩 차원: {embedding_dim}")
        
        self.embeddings_df = combined_df
        
        return combined_df
    
    def extract_embeddings(self) -> Optional[np.ndarray]:
        """
        데이터프레임에서 임베딩 배열 추출
        
        Returns:
            임베딩 배열 또는 None (실패시)
        """
        if self.embeddings_df is None:
            print("❌ 임베딩 데이터가 없습니다. 먼저 파일을 로드해주세요.")
            return None
        
        print("🔄 임베딩 배열 추출 중...")
        
        try:
            # 임베딩 리스트를 numpy 배열로 변환
            embeddings_list = self.embeddings_df['qwen3_embedding'].tolist()
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            
            print(f"✅ 임베딩 배열 추출 완료: {embeddings_array.shape}")
            
            self.embeddings = embeddings_array
            
            return embeddings_array
            
        except Exception as e:
            print(f"❌ 임베딩 배열 추출 실패: {e}")
            return None
    
    def perform_clustering(self, algorithm: str = 'kmeans', **kwargs) -> Optional[np.ndarray]:
        """
        클러스터링 수행
        
        Args:
            algorithm: 클러스터링 알고리즘 ('kmeans', 'hdbscan')
            **kwargs: 알고리즘별 파라미터
            
        Returns:
            클러스터 라벨 배열 또는 None (실패시)
        """
        if self.embeddings is None:
            print("❌ 임베딩이 없습니다. 먼저 임베딩을 추출해주세요.")
            return None
        
        print(f"🎯 클러스터링 수행: {algorithm.upper()}")
        print(f"📊 임베딩 형태: {self.embeddings.shape}")
        
        try:
            if algorithm.lower() == 'kmeans':
                n_clusters = kwargs.get('n_clusters', 8)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = clusterer.fit_predict(self.embeddings)
                
                print(f"📈 K-means 클러스터링 완료 (k={n_clusters})")
                
            elif algorithm.lower() == 'hdbscan':
                min_cluster_size = kwargs.get('min_cluster_size', 15)
                min_samples = kwargs.get('min_samples', 5)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                cluster_labels = clusterer.fit_predict(self.embeddings)
                
                print(f"📈 HDBSCAN 클러스터링 완료 (min_size={min_cluster_size})")
                
            else:
                print(f"❌ 지원하지 않는 알고리즘: {algorithm}")
                return None
            
            # 클러스터링 결과 분석
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            print(f"📊 클러스터링 결과:")
            print(f"   - 클러스터 수: {n_clusters}")
            print(f"   - 노이즈 포인트: {n_noise}")
            print(f"   - 전체 포인트: {len(cluster_labels)}")
            
            # 실루엣 스코어 계산 (노이즈 포인트 제외)
            if n_clusters > 1:
                mask = cluster_labels != -1
                if np.sum(mask) > 1:
                    silhouette_avg = silhouette_score(self.embeddings[mask], cluster_labels[mask])
                    print(f"   - 실루엣 스코어: {silhouette_avg:.3f}")
            
            self.cluster_labels = cluster_labels
            
            return cluster_labels
            
        except Exception as e:
            print(f"❌ 클러스터링 실패: {e}")
            return None
    
    def analyze_clusters(self) -> dict:
        """클러스터 결과 분석"""
        if self.cluster_labels is None or self.embeddings_df is None:
            print("❌ 클러스터링 결과가 없습니다.")
            return {}
        
        print("📈 클러스터 분석 중...")
        
        analysis = {}
        unique_labels = set(self.cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 노이즈 클러스터
                continue
            
            cluster_mask = self.cluster_labels == label
            cluster_data = self.embeddings_df[cluster_mask]
            
            # 클러스터별 샘플 문서
            sample_titles = cluster_data['title'].head(5).tolist() if 'title' in cluster_data.columns else []
            
            analysis[label] = {
                'size': int(np.sum(cluster_mask)),
                'sample_titles': sample_titles
            }
        
        self.results['cluster_analysis'] = analysis
        
        # 클러스터별 크기 및 비율 출력
        total_docs = len(self.cluster_labels)
        print(f"📊 클러스터별 문서 분포:")
        for label, info in analysis.items():
            percentage = (info['size'] / total_docs) * 100
            print(f"   클러스터 {label}: {info['size']}개 문서 ({percentage:.1f}%)")
        
        # 노이즈 포인트도 출력
        noise_count = list(self.cluster_labels).count(-1)
        if noise_count > 0:
            noise_percentage = (noise_count / total_docs) * 100
            print(f"   노이즈: {noise_count}개 문서 ({noise_percentage:.1f}%)")
        
        return analysis
    
    def reduce_dimensions(self, n_components: int = 2) -> Optional[np.ndarray]:
        """차원 축소 (시각화용)"""
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
            reduced_embeddings = np.asarray(reduced_embeddings, dtype=np.float32)
            
            self.reduced_embeddings = reduced_embeddings
            
            print(f"✅ 차원 축소 완료: {reduced_embeddings.shape}")
            return reduced_embeddings
            
        except Exception as e:
            print(f"❌ 차원 축소 실패: {e}")
            return None
    
    def visualize_clusters(self, save_path: Optional[str] = None):
        """클러스터링 결과 시각화"""
        if self.reduced_embeddings is None:
            print("🔄 차원 축소가 필요합니다...")
            if self.reduce_dimensions() is None:
                print("❌ 차원 축소 실패로 시각화를 중단합니다.")
                return
        
        if self.cluster_labels is None:
            print("❌ 클러스터링이 수행되지 않았습니다.")
            return
        
        print("🎨 클러스터링 결과 시각화 중...")
        
        try:
            # Matplotlib 시각화
            plt.figure(figsize=(12, 8))
            
            # 클러스터별 색상 지정
            unique_labels = set(self.cluster_labels)
            colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:  # 노이즈 포인트
                    color = 'black'
                    marker = 'x'
                    alpha = 0.5
                else:
                    marker = 'o'
                    alpha = 0.7
                
                mask = self.cluster_labels == label
                plt.scatter(
                    self.reduced_embeddings[mask, 0],
                    self.reduced_embeddings[mask, 1],
                    c=[color],
                    marker=marker,
                    alpha=alpha,
                    s=50,
                    label=f'Cluster {label}' if label != -1 else 'Noise'
                )
            
            plt.title('Document Clustering Results (UMAP Dimensionality Reduction)', fontsize=16)
            plt.xlabel('UMAP Component 1', fontsize=12)
            plt.ylabel('UMAP Component 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 시각화 저장: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 시각화 실패: {e}")
    
    def save_clustering_results(self, output_path: str = "./clustering_results.csv"):
        """
        클러스터링 결과를 원본 데이터에 추가하여 저장
        
        Args:
            output_path: 저장할 파일 경로
        """
        if self.embeddings_df is None or self.cluster_labels is None:
            print("❌ 데이터 또는 클러스터링 결과가 없습니다.")
            return
        
        print("💾 클러스터링 결과 저장 중...")
        
        # 원본 데이터프레임 복사
        result_df = self.embeddings_df.copy()
        
        # 클러스터링 결과 컬럼 추가
        result_df['cluster'] = self.cluster_labels
        
        # 임베딩 컬럼 제거 (용량 절약)
        if 'qwen3_embedding' in result_df.columns:
            result_df = result_df.drop('qwen3_embedding', axis=1)
        
        # 클러스터링 통계 정보 출력 (비율 포함)
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        total_docs = len(self.cluster_labels)
        
        print(f"📊 클러스터별 문서 분포:")
        for cluster_id, count in cluster_counts.items():
            percentage = (count / total_docs) * 100
            if cluster_id == -1:
                print(f"   노이즈: {count}개 ({percentage:.1f}%)")
            else:
                print(f"   클러스터 {cluster_id}: {count}개 ({percentage:.1f}%)")
        
        # CSV 파일로 저장
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 클러스터링 결과 저장 완료: {output_path}")
        print(f"📋 저장된 컬럼: {list(result_df.columns)}")
        
        return result_df
    
    def save_results(self, output_dir: str = "./clustering_results"):
        """전체 결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 클러스터링 결과 저장
        if self.cluster_labels is not None:
            csv_path = output_path / f"embedding_clustering_results_{timestamp}.csv"
            self.save_clustering_results(str(csv_path))
        
        # 시각화 저장
        if self.reduced_embeddings is not None:
            viz_path = output_path / f"embedding_cluster_visualization_{timestamp}.png"
            self.visualize_clusters(save_path=str(viz_path))

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='사전 생성된 임베딩을 이용한 클러스터링')
    parser.add_argument('--input-dir', default='./embeddings_output', help='임베딩 파일 디렉토리')
    parser.add_argument('--file-format', default='csv', choices=['csv', 'pickle'], 
                       help='임베딩 파일 형식')
    parser.add_argument('--algorithm', default='kmeans', choices=['kmeans', 'hdbscan'], 
                       help='클러스터링 알고리즘')
    parser.add_argument('--n-clusters', type=int, default=6, help='클러스터 수 (K-means용)')
    parser.add_argument('--min-cluster-size', type=int, default=6, help='최소 클러스터 크기 (HDBSCAN용)')
    parser.add_argument('--output-dir', default='./clustering_results', help='결과 저장 디렉토리')
    parser.add_argument('--output-file', default='./clustering_results/clustering_results.csv', help='클러스터링 결과 CSV 파일')
    parser.add_argument('--no-visualization', action='store_true', help='시각화 건너뛰기')
    
    args = parser.parse_args()
    
    print("🚀 임베딩 기반 클러스터링 시작!")
    
    # 클러스터링 시스템 초기화
    clustering_system = EmbeddingClusteringSystem()
    
    # 1. 임베딩 파일 로드
    embeddings_df = clustering_system.load_embedding_files(args.input_dir, args.file_format)
    
    if embeddings_df is None:
        print("❌ 임베딩 파일 로드 실패")
        return
    
    # 2. 임베딩 배열 추출
    embeddings = clustering_system.extract_embeddings()
    
    if embeddings is None:
        print("❌ 임베딩 배열 추출 실패")
        return
    
    # 3. 클러스터링 수행
    if args.algorithm == 'kmeans':
        cluster_labels = clustering_system.perform_clustering('kmeans', n_clusters=args.n_clusters)
    else:
        cluster_labels = clustering_system.perform_clustering('hdbscan', min_cluster_size=args.min_cluster_size)
    
    if cluster_labels is None:
        print("❌ 클러스터링 실패")
        return
    
    # 4. 결과 분석
    clustering_system.analyze_clusters()
    
    # 5. 시각화
    if not args.no_visualization:
        clustering_system.visualize_clusters()
    
    # 6. 결과 저장
    clustering_system.save_results(args.output_dir)
    clustering_system.save_clustering_results(args.output_file)
    
    print("🎉 임베딩 기반 클러스터링 완료!")

if __name__ == "__main__":
    main() 