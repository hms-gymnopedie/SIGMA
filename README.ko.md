# SIGMA: Site Imagery to Geometric Map Automation

**SIGMA**는 산업 현장에서 촬영된 영상으로부터 2D 평면도와 3D 디지털 트윈을 자동으로 생성하는 엔드투엔드 프레임워크입니다.

## 주요 기능

- **자동 프레임 추출**: 영상에서 최적의 이미지 시퀀스를 자동으로 추출합니다 (블러/중복 제거 포함).
- **고품질 3D 복원**: 3D Gaussian Splatting을 활용하여 사실적인 3D 모델을 생성합니다.
- **2D 평면도 생성**: 3D 데이터로부터 Occupancy Map 기반 2D 평면도를 자동 생성합니다.
- **하이브리드 아키텍처**: 전처리/맵 생성은 로컬 Mac에서, GPU 연산(COLMAP, 학습)은 서버에서 수행합니다.

## 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| 카메라 포즈 추정 | COLMAP (pycolmap) |
| 3D 복원 | 3D Gaussian Splatting (gsplat) |
| 2D 맵 생성 | BEV 투영 + Occupancy Grid (OpenCV) |
| 3D 내보내기 | Open3D (PLY, OBJ) |
| CLI | Typer |
| 설정 관리 | Pydantic + YAML |

## 설치

> `uv`를 사용하여 가상환경을 생성합니다 (`brew install uv` 또는 `pip install uv`).

### 로컬 환경 (Mac/PC)

```bash
uv venv
source .venv/bin/activate
uv pip install -e "."
```

### 서버 환경 (GPU)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[server]"
```

### 개발 환경

```bash
uv pip install -e ".[dev]"
```

## 프로젝트 구조

```
sigma/
├── configs/
│   └── default.yaml          # 기본 설정 파일
├── scripts/
│   ├── run_colmap.sh         # 서버용 COLMAP 실행 스크립트
│   └── run_train.sh          # 서버용 3DGS 학습 스크립트
├── src/sigma/
│   ├── cli.py                # CLI 엔트리 포인트
│   ├── config.py             # Pydantic 설정 모델
│   ├── preprocessor/
│   │   └── video.py          # 영상 전처리 (프레임 추출, 필터링)
│   ├── sfm/
│   │   └── colmap_runner.py  # COLMAP SfM 파이프라인
│   ├── gaussian_splatting/
│   │   ├── trainer.py        # 3DGS 학습기
│   │   └── converter.py      # 3DGS → 포인트 클라우드 변환
│   ├── map_generator/
│   │   ├── map_2d.py         # 2D 평면도 생성기
│   │   └── map_3d.py         # 3D 모델 내보내기
│   └── visualization/
│       └── viewer.py         # 결과 시각화
├── tests/                    # 단위 테스트
├── MANUAL_GUIDE.md           # 실행 가이드 (영문)
├── MANUAL_GUIDE.ko.md        # 실행 가이드 (한글)
└── pyproject.toml            # 패키지 설정
```

## 빠른 시작

```bash
# 1. 로컬: 프레임 추출
sigma extract-frames --video input.mp4 --output ./frames

# 2. (수동) 프레임을 서버로 업로드
rsync -avz ./frames/ user@server:~/workspace/frames/

# 3. 서버: COLMAP 실행
python -m sigma.sfm.colmap_runner --image_dir ./frames --output_dir ./colmap_out

# 4. 서버: 3DGS 학습
python -m sigma.gaussian_splatting.trainer --colmap_dir ./colmap_out --output_dir ./gs_model

# 5. (수동) 모델을 로컬로 다운로드
rsync -avz user@server:~/workspace/gs_model/ ./gs_model/

# 6. 로컬: 맵 생성
sigma generate-map --model ./gs_model/point_cloud.ply --output ./maps
```

자세한 실행 방법은 [실행 가이드 (한글)](MANUAL_GUIDE.ko.md)를 참고하세요.

## 테스트

```bash
pytest tests/
```

## 라이선스

MIT License
