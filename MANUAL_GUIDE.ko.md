# SIGMA 실행 가이드

이 문서는 SIGMA 파이프라인을 **로컬 Mac**과 **학교 서버(GPU)** 간에 나누어 실행하는 방법을 단계별로 설명합니다.

---

## 사전 준비

> **참고**: 의존성 충돌을 방지하기 위해 `uv`를 사용하여 가상환경을 생성합니다.

### uv 설치 (아직 없다면)
```bash
# Mac
brew install uv
# 또는
pip install uv
```

### 로컬 Mac
```bash
cd sigma
uv venv                       # .venv 가상환경 생성
source .venv/bin/activate     # 가상환경 활성화
uv pip install -e "."         # 로컬 의존성 설치
```

### 학교 서버 (GPU)
```bash
# 서버에 sigma 소스 복사 후
cd sigma
uv venv
source .venv/bin/activate
uv pip install -e ".[server]"   # 서버 의존성 설치 (pycolmap, gsplat, torch 등)

# 실행 스크립트에 실행 권한 부여
chmod +x scripts/run_colmap.sh scripts/run_train.sh
```

### SSH 접속 확인
```bash
ssh user@서버주소
```

---

## 파이프라인 단계

### 1단계. 로컬: 영상에서 프레임 추출

**Mac**에서 실행합니다.

```bash
sigma extract-frames \
    --video /path/to/영상파일.mp4 \
    --output ./my_project/frames
```

**결과**: `frames/` 디렉토리에 `frame_00000.jpg`, `frame_00001.jpg`, ... 형태로 이미지가 저장됩니다.

> **참고**: 기본 설정은 2 FPS로 추출하며, 블러가 심한 프레임과 중복 프레임은 자동으로 제거됩니다. 설정을 변경하려면 `configs/default.yaml`을 수정하세요.

---

### 2단계. 전송: 프레임을 서버로 업로드

`rsync` 또는 `scp`를 사용하여 추출한 프레임을 서버로 업로드합니다.

```bash
# rsync 사용 (권장 - 진행률 표시, 이어받기 지원)
rsync -avz --progress \
    ./my_project/frames/ \
    user@서버주소:~/sigma_workspace/project_001/frames/

# 또는 scp 사용
scp -r ./my_project/frames/ \
    user@서버주소:~/sigma_workspace/project_001/frames/
```

---

### 3단계. 서버: COLMAP 실행 (Structure-from-Motion)

**서버**에 SSH 접속한 후 실행합니다.

```bash
ssh user@서버주소
cd ~/sigma_workspace/sigma    # sigma 소스 디렉토리로 이동
```

COLMAP을 실행합니다:

```bash
./scripts/run_colmap.sh \
    ~/sigma_workspace/project_001/frames \
    ~/sigma_workspace/project_001/colmap_out
```

**결과**: `colmap_out/sparse/0/` 에 카메라 포즈 및 sparse 포인트 클라우드가 저장됩니다.

> **소요 시간**: 프레임 수에 따라 10분 ~ 수 시간이 걸릴 수 있습니다.

---

### 4단계. 서버: 3D Gaussian Splatting 학습

서버에서 계속 실행합니다:

```bash
./scripts/run_train.sh \
    ~/sigma_workspace/project_001/colmap_out \
    ~/sigma_workspace/project_001/gs_model
```

**결과**: `gs_model/point_cloud.ply`에 학습된 3DGS 모델이 저장됩니다.

> **팁**: `tmux` 또는 `screen`을 사용하면 SSH 연결이 끊겨도 학습이 계속됩니다.
> ```bash
> tmux new -s sigma_train
> # 학습 명령 실행
> # Ctrl+B, D로 세션 분리
> # 나중에 tmux attach -t sigma_train 으로 복귀
> ```

> **소요 시간**: 기본 30,000 이터레이션 기준, RTX 4090에서 약 15~30분 소요됩니다.

---

### 5단계. 전송: 학습 모델을 로컬로 다운로드

**Mac**에서 실행합니다:

```bash
rsync -avz --progress \
    user@서버주소:~/sigma_workspace/project_001/gs_model/ \
    ./my_project/gs_model/
```

---

### 6단계. 로컬: 맵 생성

**Mac**에서 실행합니다:

```bash
sigma generate-map \
    --model ./my_project/gs_model/point_cloud.ply \
    --output ./my_project/maps
```

**결과물**:
| 파일 | 설명 |
|------|------|
| `maps/occupancy_map.png` | 2D 평면도 (Occupancy Grid) |
| `maps/model_export.ply` | 3D 포인트 클라우드 / 메쉬 |

---

### 7단계. 결과 확인

```bash
# 2D 평면도 보기
sigma view-2d --map ./my_project/maps/occupancy_map.png

# 3D 모델 보기
sigma view-3d --model ./my_project/maps/model_export.ply
```

---

## 설정 변경

설정 파일 `configs/default.yaml`을 수정하여 동작을 변경할 수 있습니다:

```yaml
preprocessor:
  extraction_fps: 2        # 초당 추출 프레임 수
  blur_threshold: 100.0    # 블러 감지 임계값 (낮을수록 엄격)
  dedup_threshold: 5       # 중복 감지 임계값 (낮을수록 엄격)

sfm:
  matching_type: "sequential"  # 매칭 방식: sequential, exhaustive, vocab_tree
  use_gpu: true                # GPU 가속 사용

gaussian_splatting:
  iterations: 30000            # 학습 반복 횟수 (높을수록 품질↑, 시간↑)

map_2d:
  z_slice_min: 0.5             # 높이 슬라이스 최소값 (미터)
  z_slice_max: 2.0             # 높이 슬라이스 최대값 (미터)
  resolution: 0.05             # 그리드 해상도 (미터/픽셀)
```

---

## 문제 해결

### pycolmap 설치가 안 될 때
```bash
# conda 사용 시
conda install -c conda-forge pycolmap

# pip 사용 시 (CUDA 필요)
pip install pycolmap
```

### COLMAP 실행 시 GPU 메모리 부족
`configs/default.yaml`에서 `use_gpu: false`로 변경하세요 (속도가 느려집니다).

### 평면도 품질이 낮을 때
- `z_slice_min` / `z_slice_max` 값을 조정하여 관심 높이 범위를 좁혀보세요.
- `resolution` 값을 낮추면 (예: `0.02`) 더 세밀한 맵이 생성됩니다.
- `occupancy_threshold` 값을 높이면 노이즈가 줄어듭니다.

### 3DGS 학습이 느릴 때
- `iterations`를 줄여보세요 (예: `7000`으로 먼저 테스트).
- `CUDA_VISIBLE_DEVICES=0` 환경변수로 특정 GPU만 사용할 수 있습니다.
- 두 GPU를 모두 사용하려면: `CUDA_VISIBLE_DEVICES=0,1`

---

## 최신 업데이트 (2026-02-14)

### 코드 품질 개선 사항

다음 버그들이 수정되었습니다:

1. **3DGS 모델 변환 안정성 향상** (`converter.py`)
   - Sigmoid 연산 오버플로우 방지 (매우 큰 음수 값 처리)
   - PLY 파일에 `opacity` 또는 `f_dc_*` 필드가 없어도 정상 동작

2. **2D 맵 생성 정확도 향상** (`map_2d.py`)
   - 불필요한 코드 제거 (미사용 인덱스 계산)
   - Histogram range 계산 오류 수정 → BEV 투영 정합성 개선

3. **비디오 프레임 추출 안정성** (`video.py`)
   - FPS가 0인 잘못된 비디오 파일 감지 및 에러 메시지 개선
   - Division by zero 방지

4. **3D 모델 뷰어 안정성** (`viewer.py`)
   - 존재하지 않는 파일 로드 시 명확한 에러 메시지 출력

자세한 변경 이력은 `sigma/CHANGELOG.md`를 참조하세요.
