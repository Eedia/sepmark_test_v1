# SepMark 디코더

이미지와 비디오에 숨겨진 워터마크를 추출하는 디코더 모듈.

## 주요 기능

- **이미지 워터마크 추출**: 워터마킹된 이미지에서 숨겨진 메시지 추출
- **비디오 워터마크 분석**: 프레임별 워터마크 추출 및 진위 여부 판별
- **두 가지 디코더**: Tracer(강건한 디코더)와 Detector(준강건한 디코더) 모델 제공
- **시각화 도구**: 추출 결과에 대한 이미지 및 비디오 시각화 제공

## 시스템 요구사항

- Python 3.7 이상
- CUDA 지원 GPU (선택 사항이지만 권장)
- FFmpeg (비디오 처리용)

## 설치 방법

1. 저장소 클론 또는 다운로드:
```
git clone https://github.com/Eedia/sepmark_test_v1.git
cd sepmark/decoder
```

2. 필요한 패키지 설치:
```
pip install -r requirements.txt
```

3. FFmpeg 설치:
   - Windows: [FFmpeg 공식 사이트](https://ffmpeg.org/download.html)에서 다운로드하여 PATH에 추가
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

## 사용 방법

### 1. 모델 학습

```
python main.py train tracer     # Tracer 모델 학습
python main.py train detector   # Detector 모델 학습
python main.py train all        # 두 모델 모두 학습
```

### 2. 이미지 워터마크 추출

```
python main.py eval [path/to/image.jpg]
```

### 3. 비디오 워터마크 분석

```
python main.py video [path/to/video.mp4] [--interval 10] [--no-output]
```

- `--interval N`: N 프레임마다 분석 (기본값: 10)
- `--no-output`: 분석 결과 비디오를 생성하지 않음

## 결과 해석

분석 결과는 다음과 같이 해석:

- **Tracer 정확도**: 강건한 디코더가 추출한 워터마크와 원본 메시지의 일치도
- **Detector 정확도**: 준강건한 디코더가 추출한 워터마크와 원본 메시지의 일치도
- **불일치율(Mismatch)**: 두 디코더 간의 추출 결과 차이 (40% 미만이면 진짜로 판별)

## 프로젝트 구조

```
decoder/
├── Decoder.py              # 디코더 모델 정의
├── Decoder_train.py        # 모델 학습 코드
├── Decoder_evaluate.py     # 이미지 평가 코드
├── Video_Decoder.py        # 비디오 처리 코드
├── Noise_Layer.py          # 노이즈 레이어 정의
├── ffmpeg_utils.py         # FFmpeg 유틸리티 함수
├── main.py                 # 메인 실행 스크립트
├── saved_models/           # 저장된 모델 파일
│   ├── tracer_model.pth
│   └── detector_model.pth
├── eval_results/           # 이미지 평가 결과
├── video_results/          # 비디오 평가 결과
├── output_frames/          # 출력 프레임
└── temp_frames/            # 임시 처리 프레임
```

## 문제 해결

### CUDA 오류

CUDA 관련 오류가 발생하는 경우:

1. CUDA 드라이버와 PyTorch 버전이 호환되는지 확인
2. GPU 메모리 부족 시 배치 크기 감소 또는 CPU 모드로 전환
3. CUDA 관련 옵션 제거: 예를 들어 FFmpeg 명령에서 `-hwaccel cuda` 옵션 제거 ## 현재는 cpu로만 ffmpeg를 사용하기 떄문에 괜찮

### FFmpeg 오류

FFmpeg 관련 오류가 발생하는 경우:

1. FFmpeg가 정상적으로 설치되어 있는지 확인 (명령줄에서 `ffmpeg -version` 실행)
2. 파일 경로에 공백이나 특수 문자가 있는 경우 따옴표로 경로 감싸기
3. 인코더/디코더 포맷 호환성 문제: `-hwaccel_output_format cuda` 대신 CPU 기반 인코딩 사용
