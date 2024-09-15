import torch
import re
import numpy as np

def analyze_diary(model, tokenizer, diary_text):
    model.eval()

    # 문장 분할 (마침표, 느낌표, 물음표, 세미콜론, 쉼표로 문장을 구분)
    sentences = re.split(r'(?<=[.!?,;])\s+', diary_text.strip())

    # 감정 목록 및 확률 초기화
    emotions = ["불안", "당황", "분노", "슬픔", "중립", "행복"]
    total_probabilities = np.zeros(len(emotions))  # 감정별 총합 확률

    # 문장별 감정 분석
    for idx, sentence in enumerate(sentences):
        if sentence:  # 빈 문장은 제외
            # print(f"\nAnalyzing Sentence {idx + 1}: {sentence}")

            encoding = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=128,
                return_attention_mask=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # 문장별 감정 확률 계산
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            intensity = probabilities.squeeze().cpu().numpy()

            # 감정 확률을 총합에 더함
            total_probabilities += intensity

    # 중립(4번째 감정)을 제외한 감정 확률을 계산
    non_neutral_emotions = [(emotions[i], total_probabilities[i]) for i in range(len(emotions)) if i != 4]

    # 감정 확률을 정규화하여 100%로 만듦
    non_neutral_probabilities = [prob for _, prob in non_neutral_emotions]
    normalized_probabilities = np.array(non_neutral_probabilities) / sum(non_neutral_probabilities) * 100

    # 상위 3개의 감정을 추출(5퍼센트 미만 감정은 제외)
    top_3_indices = np.argsort(normalized_probabilities)[-3:][::-1]
    top_3_emotions = [(non_neutral_emotions[i][0], normalized_probabilities[i]) for i in top_3_indices if normalized_probabilities[i] >= 5]

    # 결과 반환
    return top_3_emotions