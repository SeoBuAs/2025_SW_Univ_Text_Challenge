#!/usr/bin/env python3
import re
import random
import time
import os
import argparse
from datetime import datetime

from ollama import Client
import pandas as pd
import json
from typing import List
from tqdm import tqdm

client = Client(host='http://localhost:11434')

MODEL_LIST = [
    'benedict/linkbricks-llama3.1-korean:8b',   
    'gemma3:27b-it-qat',                          
    'exaone3.5:32b-instruct-q4_K_M',            
    'mistral-small3.2:24b-instruct-2506-q4_K_M',
    'qwen2.5:32b-instruct-q4_K_M',
    'exaone3.5:32b-instruct-q4_K_M',
]

system_prompt = """
You are helping to create synthetic text data for AI detection training.

You will be given:
1. DOCUMENT CONTEXT - The overall content and theme of the full document
2. PREVIOUS PARAGRAPH - The paragraph that comes before the target paragraph
3. TARGET PARAGRAPH - The paragraph that needs to be transformed/rewritten
4. NEXT PARAGRAPH - The paragraph that comes after the target paragraph

Your task is to REWRITE/TRANSFORM the TARGET paragraph while:
- Maintaining the same core meaning and information
- Keeping consistency with the overall document context
- Ensuring smooth transitions with previous and next paragraphs
- Using different wording, sentence structure, and expression style
- Preserving the same paragraph length (similar number of sentences)
- Maintaining the same tone and register as the original document

# Transformation Guidelines:
- Paraphrase the content using different vocabulary and sentence structures
- Keep the same key information and facts
- Maintain logical flow and coherence with surrounding paragraphs
- Use synonyms, different expressions, and varied sentence patterns
- Preserve any specific names, dates, numbers, or technical terms mentioned
- Match the writing style and complexity level of the original document
- For Korean text: Use natural Korean expressions and maintain formality level
- Keep approximately the same length as the original paragraph

# Requirements:
- The transformed paragraph should convey the same information as the original
- Use different linguistic expressions while maintaining semantic equivalence
- Ensure natural flow with the previous and next paragraphs
- Preserve document-specific terminology and proper nouns
- Maintain consistency with the overall document style and tone

Return only JSON: {"transformed_paragraph": "your rewritten paragraph here"}
"""

def get_model_for_index(index: int, model_switch_interval: int = 10) -> str:
    """인덱스에 따라 순환하는 모델 선택"""
    model_cycle = (index // model_switch_interval) % len(MODEL_LIST)
    return MODEL_LIST[model_cycle]

def smart_split_text_into_paragraphs(text: str, target_paragraph_length: int = 200) -> List[str]:
    """긴 텍스트를 지능적으로 문단으로 분할"""
    
    # 1단계: 이미 \n\n으로 구분된 문단이 있는지 확인
    existing_paragraphs = re.split(r'\n\s*\n', text.strip())
    existing_paragraphs = [p.strip() for p in existing_paragraphs if p.strip()]

    if len(existing_paragraphs) > 1:
        return existing_paragraphs

    # 2단계: 단일 텍스트인 경우 문장 기반으로 분할
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 2:
        return [text]

    # 3단계: 문장들을 적절한 길이의 문단으로 그룹화
    paragraphs = []
    current_paragraph = ""

    for sentence in sentences:
        potential_length = len(current_paragraph) + len(sentence) + 1

        if current_paragraph and potential_length > target_paragraph_length:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = sentence
        else:
            if current_paragraph:
                current_paragraph += " " + sentence
            else:
                current_paragraph = sentence

    if current_paragraph.strip():
        paragraphs.append(current_paragraph.strip())

    return paragraphs

def transform_paragraph(target_paragraph: str, prev_paragraph: str, next_paragraph: str, 
                       full_document: str, model_name: str, max_retries: int = 3) -> str:
    """특정 문단을 변형/재작성"""
    
    # 전체 문서가 너무 길면 요약본 사용
    document_context = full_document
    if len(full_document) > 2000:
        document_context = full_document[:300] + "...[중간 내용 생략]..." + full_document[-300:]

    # 너무 긴 문단들은 일부만 사용
    if prev_paragraph and len(prev_paragraph) > 500:
        prev_paragraph = prev_paragraph[-400:]
    if next_paragraph and len(next_paragraph) > 500:
        next_paragraph = next_paragraph[:400:]

    # 프롬프트 생성
    user_prompt = f"""DOCUMENT CONTEXT (for consistency):
{document_context}

PREVIOUS PARAGRAPH:
{prev_paragraph if prev_paragraph else "[Document Beginning]"}

TARGET PARAGRAPH (to be transformed):
{target_paragraph}

NEXT PARAGRAPH:
{next_paragraph if next_paragraph else "[Document End]"}

Please REWRITE/TRANSFORM the TARGET paragraph while maintaining the same core meaning and information, but using different wording and sentence structures. Keep consistency with the surrounding context."""

    for attempt in range(max_retries):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                think=False,
            )

            # JSON 추출
            json_match = re.search(r'\{.*?\}', response.message.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                transformed = data.get("transformed_paragraph", "").strip()
                if transformed and len(transformed) > 30:  # 최소 길이 체크
                    return transformed

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)

    return None

def process_single_document_transform_one(document: str, model_name: str) -> List[dict]:
    """단일 문서에서 중간 문단 1개를 선택해서 변형"""
    
    # 1단계: 문단 분할
    paragraphs = smart_split_text_into_paragraphs(document, target_paragraph_length=300)
    
    if len(paragraphs) < 2:  # 최소 2개 문단은 있어야 함
        return []

    # 2단계: 중간 문단 1개 랜덤 선택 (첫 번째와 마지막 제외)
    if len(paragraphs) <= 2:
        # 2개뿐이면 둘 중 하나 선택
        target_idx = random.choice([0, 1])
    else:
        # 3개 이상이면 중간 문단들 중에서 선택 (첫 번째, 마지막 제외)
        middle_indices = list(range(1, len(paragraphs) - 1))
        target_idx = random.choice(middle_indices)
    
    # 이전, 현재, 다음 문단 설정
    prev_para = paragraphs[target_idx - 1] if target_idx > 0 else None
    target_para = paragraphs[target_idx]
    next_para = paragraphs[target_idx + 1] if target_idx < len(paragraphs) - 1 else None
    
    # 문단 변형
    transformed_paragraph = transform_paragraph(
        target_para, prev_para, next_para, document, model_name
    )
    
    if transformed_paragraph:
        # 변형된 문단으로 전체 문서 재구성
        new_paragraphs = paragraphs.copy()
        new_paragraphs[target_idx] = transformed_paragraph
        
        transformed_document = '\n\n'.join(new_paragraphs)
        
        return [{
            'transformed_document': transformed_document,
            'original_paragraph': target_para,
            'transformed_paragraph': transformed_paragraph,
            'transformed_position': target_idx,
            'model_used': model_name,
            'original_length': len(document),
            'transformed_length': len(transformed_document),
            'total_paragraphs': len(paragraphs)
        }]
    
    return []

def batch_transform_data(input_csv: str, output_csv: str, num_samples: int, 
                        model_switch_interval: int = 10, save_interval: int = 0, 
                        verbose: bool = False):
    """배치로 데이터 변형 처리"""

    print(f"Loading data from {input_csv}")

    # 데이터 로드
    df = pd.read_csv(input_csv)

    # Human 텍스트만 선택 (generated == 0)
    human_texts = df[df['generated'] == 0].copy().reset_index(drop=True)

    if len(human_texts) == 0:
        print("Error: No human texts found (generated == 0)")
        return

    print(f"Total data: {len(df)}, Human texts: {len(human_texts)}")

    # 처리할 샘플 수 제한
    if num_samples > len(human_texts):
        print(f"Warning: Requested {num_samples} samples, but only {len(human_texts)} available")
        num_samples = len(human_texts)

    # 랜덤 샘플 선택
    selected_indices = random.sample(range(len(human_texts)), num_samples)
    selected_samples = human_texts.iloc[selected_indices].copy()

    # 원본 인덱스 정보 보존
    original_indices = [human_texts.index[i] for i in selected_indices]

    print(f"Processing {len(selected_samples)} samples")
    print(f"Each document will generate 1 variant (middle paragraph transformation)")
    print(f"Models: {MODEL_LIST}")
    print(f"Model switch interval: every {model_switch_interval} samples")
    if save_interval > 0:
        print(f"Intermediate save interval: every {save_interval} samples")
    print(f"Output file: {output_csv}")

    # 결과 저장을 위한 리스트
    transformed_data = []
    success_count = 0
    fail_count = 0

    # 진행 상황 추적
    for idx in tqdm(range(len(selected_samples)), desc="Processing documents"):
        try:
            row = selected_samples.iloc[idx]
            original_row_number = original_indices[idx]

            # 모델 선택
            model_name = get_model_for_index(idx, model_switch_interval)

            document = row['full_text']
            original_title = row.get('title', f'{idx}')

            if verbose:
                print(f"Processing sample {idx} (original row {original_row_number}) with model {model_name}")

            # 문서의 중간 문단 1개 변형
            results = process_single_document_transform_one(document, model_name)

            if results:
                for i, result in enumerate(results):
                    transformed_row = {
                        'title': original_title,
                        'original_text': document,
                        'original_document_number': original_row_number,
                        'original_paragraph': result['original_paragraph'],
                        'transformed_paragraph': result['transformed_paragraph'],
                        'transformed_position': result['transformed_position'],
                        'merge_text': result['transformed_document'],
                        'generated': 1,
                        'model_use': result['model_used']
                    }

                    transformed_data.append(transformed_row)
                    success_count += 1
                
                if verbose:
                    print(f"  Generated 1 variant by transforming paragraph {results[0]['transformed_position'] + 1}")
            else:
                fail_count += 1
                if verbose:
                    print(f"Failed to process sample {idx}")

            # 중간 저장 체크
            if save_interval > 0 and (idx + 1) % save_interval == 0 and transformed_data:
                try:
                    temp_filename = output_csv.replace('.csv', '_temp.csv')
                    if os.path.exists(output_csv):
                        import shutil
                        shutil.copy2(output_csv, temp_filename)

                    temp_df = pd.DataFrame(transformed_data)
                    temp_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                    print(f"\nIntermediate save: {output_csv} ({len(transformed_data)} records)")

                except Exception as save_error:
                    print(f"\nWarning: Intermediate save failed: {save_error}")

        except Exception as e:
            fail_count += 1
            if verbose:
                print(f"Error processing sample {idx}: {e}")
            continue

    # 최종 결과 저장
    if transformed_data:
        try:
            temp_filename = output_csv.replace('.csv', '_temp.csv')
            if os.path.exists(output_csv):
                import shutil
                shutil.copy2(output_csv, temp_filename)

            transformed_df = pd.DataFrame(transformed_data)
            transformed_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

            print(f"=== 중간 문단 변형 완료 ===")
            print(f"Output file: {output_csv}")
            print(f"Total transformed documents: {len(transformed_data)}")
            print(f"Documents processed: {len(selected_samples)}")
            print(f"Success rate: {len(transformed_data)/len(selected_samples):.1%}")

            # 사용된 모델 통계
            model_usage = transformed_df['model_use'].value_counts()
            print(f"\nModel usage:")
            for model, count in model_usage.items():
                print(f"  {model}: {count}")

        except Exception as final_save_error:
            print(f"\nError: Final save failed: {final_save_error}")

def test_single_sample(input_csv: str, verbose: bool = False):
    """단일 샘플 테스트"""
    
    print("=== 중간 문단 변형 테스트 ===")

    try:
        df = pd.read_csv(input_csv)
        human_texts = df[df['generated'] == 0]

        if len(human_texts) == 0:
            print("Error: No human texts found")
            return

        # 첫 번째 샘플 선택
        sample_doc = human_texts.iloc[0]['full_text']
        sample_title = human_texts.iloc[0].get('title', 'No title')

        print(f"Selected document: {sample_title}")
        print(f"Length: {len(sample_doc)} characters")

        # 문단 분할
        paragraphs = smart_split_text_into_paragraphs(sample_doc, target_paragraph_length=300)
        print(f"Split into {len(paragraphs)} paragraphs")

        if verbose:
            for i, para in enumerate(paragraphs):
                print(f"  {i+1}. [{len(para)} chars] {para[:80]}...")

        # 첫 번째 모델 사용
        model_name = MODEL_LIST[0]
        print(f"\nUsing model: {model_name}")

        # 중간 문단 변형 테스트
        if len(paragraphs) >= 2:
            # 중간 문단 선택 로직과 동일
            if len(paragraphs) <= 2:
                target_idx = random.choice([0, 1])
            else:
                middle_indices = list(range(1, len(paragraphs) - 1))
                target_idx = random.choice(middle_indices)
            
            target_para = paragraphs[target_idx]
            prev_para = paragraphs[target_idx - 1] if target_idx > 0 else None
            next_para = paragraphs[target_idx + 1] if target_idx < len(paragraphs) - 1 else None

            print(f"\nTransforming middle paragraph {target_idx + 1} (out of {len(paragraphs)}):")
            print(f"Original: {target_para}")

            transformed = transform_paragraph(target_para, prev_para, next_para, sample_doc, model_name)

            if transformed:
                print(f"\nTransformed: {transformed}")
                print(f"\nComparison:")
                print(f"  Original length: {len(target_para)} chars")
                print(f"  Transformed length: {len(transformed)} chars")
            else:
                print("Failed to transform paragraph")

    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='AI Text Transformation System (Single Middle Paragraph)')
    parser.add_argument('--mode', choices=['test', 'batch'], required=False, default='batch',
                       help='Mode: test (single sample) or batch (multiple samples)')
    parser.add_argument('--input', default='./data/train.csv',
                       help='Input CSV file (default: ./data/train.csv)')
    parser.add_argument('--output',
                       help='Output CSV file (auto-generated if not specified)')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of samples to process (default: 5000)')
    parser.add_argument('--model-switch-interval', type=int, default=1000,
                       help='Interval for switching models (default: 1000)')
    parser.add_argument('--save-interval', type=int, default=500,
                       help='Interval for intermediate saves (default: 500)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--models', nargs='+',
                       help='List of models to use (overrides default)')

    args = parser.parse_args()

    # 모델 리스트 업데이트
    if args.models:
        global MODEL_LIST
        MODEL_LIST = args.models
        print(f"Using custom models: {MODEL_LIST}")

    if not os.path.exists('./new_data'):
        os.makedirs('./new_data')

    if args.mode == 'test':
        test_single_sample(args.input, args.verbose)

    elif args.mode == 'batch':
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"./new_data/transformed_data_{timestamp}.csv"

        batch_transform_data(
            input_csv=args.input,
            output_csv=args.output,
            num_samples=args.samples,
            model_switch_interval=args.model_switch_interval,
            save_interval=args.save_interval,
            verbose=args.verbose
        )

if __name__ == "__main__":
    main()