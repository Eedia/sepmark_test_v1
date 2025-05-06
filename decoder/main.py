import os
import sys
import argparse

# 필요한 모듈 임포트
from Decoder_train import train_decoder
from Decoder_evaluate import evaluate_decoders
from Video_Decoder import decode_video

def main():
    """SepMark 디코더 메인 실행 함수"""
    parser = argparse.ArgumentParser(description='SepMark 워터마크 디코더')
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령어')
    
    # 학습 명령어
    train_parser = subparsers.add_parser('train', help='디코더 모델 학습')
    train_parser.add_argument('model_type', choices=['tracer', 'detector', 'all'], 
                              help='학습할 모델 유형: tracer, detector, 또는 all (두 모델 모두)')
    
    # 이미지 평가 명령어
    eval_parser = subparsers.add_parser('eval', help='이미지에서 워터마크 평가')
    eval_parser.add_argument('image_path', help='평가할 이미지 파일 경로')
    
    # 비디오 처리 명령어
    video_parser = subparsers.add_parser('video', help='비디오 워터마크 디코딩')
    video_parser.add_argument('video_path', help='처리할 비디오 파일 경로')
    video_parser.add_argument('--interval', type=int, default=10, help='프레임 샘플링 간격 (기본값: 10)')
    video_parser.add_argument('--no-output', action='store_false', dest='create_output', 
                             help='분석 결과 비디오를 생성하지 않음')
    
    # 명령행 인수 파싱
    args = parser.parse_args()
    
    # 명령어에 따른 처리
    if args.command == 'train':
        print(f"[SepMark] {args.model_type} 모델 학습 시작...")
        
        if args.model_type == 'tracer':
            train_decoder('tracer')
        elif args.model_type == 'detector':
            train_decoder('detector')
        elif args.model_type == 'all':
            print("Tracer 학습 시작...")
            train_decoder('tracer')
            print("\nDetector 학습 시작...")
            train_decoder('detector')
        
    elif args.command == 'eval':
        print(f"[SepMark] 이미지 평가 시작: {args.image_path}")
        
        if os.path.exists(args.image_path):
            evaluate_decoders(args.image_path)
        else:
            print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {args.image_path}")
        
    elif args.command == 'video':
        print(f"[SepMark] 비디오 처리 시작: {args.video_path}")
        
        if os.path.exists(args.video_path):
            decode_video(args.video_path, sample_interval=args.interval)
        else:
            print(f"⚠️ 비디오 파일을 찾을 수 없습니다: {args.video_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    # 필요한 디렉토리 생성
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./eval_results", exist_ok=True)
    os.makedirs("./video_results", exist_ok=True)
    
    # 시스템 정보 출력
    print(f"[시스템 정보] Python 버전: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"[시스템 정보] PyTorch 버전: {torch.__version__}")
        print(f"[시스템 정보] CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[시스템 정보] GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("⚠️ PyTorch가 설치되어 있지 않습니다.")
    
    # 메인 함수 실행
    if len(sys.argv) > 1:
        main()
    else:
        print("[SepMark 디코더 사용법]")
        print("다음 명령어 중 하나를 사용하세요:")
        print("1. 모델 학습:")
        print("   python main.py train tracer    - Tracer 모델 학습")
        print("   python main.py train detector  - Detector 모델 학습")
        print("   python main.py train all       - 두 모델 모두 학습")
        print("")
        print("2. 이미지 평가:")
        print("   python main.py eval <이미지_경로>  - 이미지에서 워터마크 추출 및 평가")
        print("")
        print("3. 비디오 디코딩:")
        print("   python main.py video <비디오_경로> [--interval N] [--no-output]")
        print("      --interval N: N 프레임마다 분석 (기본값: 10)")
        print("      --no-output: 분석 결과 비디오를 생성하지 않음")
