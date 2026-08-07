"""
run_hgsplat.py
──────────────
HGSplat v2 파이프라인 원샷 러너: restoration.py → train.py → render.py 순차 실행.

  Step 1  restoration.py : MWFormer 복원 → images_cleaned/ + heatmap_init/
  Step 2  train.py       : cleaned 입력 + heatmap_init 으로 LongSplat 학습
                           (--heatmap_mv 지정 시 Refinement 에서 v2 활성화)
  Step 3  render.py      : 학습된 모델 렌더링

사용 예:
    # v2 (멀티뷰 일관성 ON)
    python run_hgsplat.py -s data/grass_snow -m outputs/grass_v2 \
        --eval --mode custom --heatmap_alpha 20 --heatmap_mv

    # v1 대조군 (cleaned 입력 + H_init 만, 멀티뷰 OFF)
    python run_hgsplat.py -s data/grass_snow -m outputs/grass_v1c \
        --eval --mode custom --heatmap_alpha 20

    # og_c 대조군 (cleaned 입력만, heatmap 완전 OFF — 복원 입력 효과 분리용)
    python run_hgsplat.py -s data/grass_snow -m outputs/grass_ogc \
        --eval --mode custom --no_heatmap

    # 학습만 다시 (복원 결과 재사용)
    python run_hgsplat.py -s data/grass_snow -m outputs/grass_v2b \
        --eval --mode custom --heatmap_mv --skip_restore
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(name, cmd):
    print(f"\n{'='*70}\n[{name}] {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=CODE_DIR)
    dt = time.time() - t0
    if ret.returncode != 0:
        sys.exit(f"\n[FAIL] {name} 실패 (exit={ret.returncode}, {dt/60:.1f}분) — 파이프라인 중단")
    print(f"\n[OK] {name} 완료 ({dt/60:.1f}분)", flush=True)


def main():
    p = argparse.ArgumentParser(description="HGSplat 파이프라인 (restoration → train → render)")
    # 공통
    p.add_argument("-s", "--source_path", required=True, help="씬 폴더 (images/ 포함)")
    p.add_argument("-m", "--model_path", required=True, help="학습 출력 폴더")
    # Step 1: restoration
    p.add_argument("--images_dirname", default="images", help="원본 이미지 하위 폴더명")
    p.add_argument("--cleaned_dirname", default="images_cleaned", help="복원 이미지 하위 폴더명")
    p.add_argument("--heatmap_dirname", default="heatmap_init", help="heatmap 하위 폴더명")
    p.add_argument("--heatmap_thresh", type=float, default=0.0)
    p.add_argument("--force_restore", action="store_true", help="복원 결과가 있어도 재생성")
    # Step 2: train (자주 쓰는 것만 노출, 나머지는 --train_extra 로)
    p.add_argument("--mode", default=None, help="train.py --mode (예: custom, free)")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--seed", type=int, default=0,
                   help="train.py random seed (repeat runs should use distinct seeds)")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--heatmap_alpha", type=float, default=None)
    p.add_argument("--heatmap_norm", default=None)
    p.add_argument("--heatmap_pct", type=float, default=None)
    p.add_argument("--heatmap_floor", type=float, default=None)
    p.add_argument("--heatmap_mv", action="store_true", help="[v2] Refinement 멀티뷰 일관성 활성화")
    p.add_argument("--heatmap_mv_beta", type=float, default=None)
    p.add_argument("--heatmap_mv_ramp", type=int, default=None)
    p.add_argument("--heatmap_mv_var", action="store_true",
                   help="[v2.1] H_multi 분산 게이트(고평균·고분산=눈만 억제, 텍스처 보존)")
    p.add_argument("--heatmap_mv_std_floor", type=float, default=None)
    p.add_argument("--no_heatmap", action="store_true",
                   help="[og_c] heatmap loss 완전 OFF (cleaned 입력만 사용, 일반 L1 학습)")
    p.add_argument("--train_extra", nargs=argparse.REMAINDER, default=[],
                   help="train.py 로 그대로 전달할 추가 인자 (이 뒤의 모든 토큰)")
    # Step 3: render
    p.add_argument("--iteration", type=int, default=-1, help="render.py --iteration")
    # 단계 스킵
    p.add_argument("--skip_restore", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_render", action="store_true")
    p.add_argument("--skip_metrics", action="store_true")
    p.add_argument("--no_timestamp", action="store_true",
                   help="model_path 에 _YYMMDD_HHMMSS 를 붙이지 않음")
    args = p.parse_args()

    src = os.path.abspath(args.source_path)
    model = os.path.abspath(args.model_path)

    # 타임스탬프는 러너가 직접 부여 (train.py 에는 --no_timestamp 전달).
    # → 확정된 동일 경로가 train/render/metrics 에 모두 전달된다.
    # --skip_train 시에는 기존 폴더를 그대로 쓰는 것이므로 붙이지 않는다.
    if not args.skip_train and not args.no_timestamp:
        model = f"{model}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    print(f"[info] model_path = {model}")
    input_dir = os.path.join(src, args.images_dirname)
    cleaned_dir = os.path.join(src, args.cleaned_dirname)
    heatmap_dir = os.path.join(src, args.heatmap_dirname)
    py = sys.executable
    t_start = time.time()

    # ── Step 1: restoration ─────────────────────────────
    if args.skip_restore:
        print("[skip] Step 1 (restoration)")
    else:
        cmd = [py, os.path.join(CODE_DIR, "restoration.py"),
               "--input_dir", input_dir,
               "--cleaned_dir", cleaned_dir,
               "--heatmap_dir", heatmap_dir,
               "--heatmap_thresh", str(args.heatmap_thresh)]
        if args.force_restore:
            cmd.append("--force")
        run_step("Step 1/4 restoration", cmd)

    # ── Step 2: train ───────────────────────────────────
    if args.skip_train:
        print("[skip] Step 2 (train)")
    else:
        if not os.path.isdir(cleaned_dir):
            sys.exit(f"[error] 복원 출력이 없습니다: {cleaned_dir}\n"
                     "--skip_restore 를 빼거나 restoration.py 를 먼저 실행하세요.")
        if not args.no_heatmap and not os.path.isdir(heatmap_dir):
            sys.exit(f"[error] heatmap 폴더가 없습니다: {heatmap_dir}\n"
                     "--skip_restore 를 빼거나 restoration.py 를 먼저 실행하세요.")
        cmd = [py, os.path.join(CODE_DIR, "train.py"),
               "-s", src, "-m", model,
               "--no_timestamp",
               "--seed", str(args.seed),
               "--images", args.cleaned_dirname,
               # og_c: "none" 이면 train.py 가 heatmap loss 를 명시적으로 끔
               "--heatmap_dir", "none" if args.no_heatmap else heatmap_dir]
        if args.mode is not None:
            cmd += ["--mode", args.mode]
        if args.eval:
            cmd.append("--eval")
        if args.resolution is not None:
            cmd += ["--resolution", str(args.resolution)]
        for k in ("heatmap_alpha", "heatmap_norm", "heatmap_pct",
                  "heatmap_floor", "heatmap_mv_beta", "heatmap_mv_ramp",
                  "heatmap_mv_std_floor"):
            v = getattr(args, k)
            if v is not None:
                cmd += [f"--{k}", str(v)]
        if args.heatmap_mv and not args.no_heatmap:
            cmd.append("--heatmap_mv")
        if args.heatmap_mv_var and not args.no_heatmap:
            cmd.append("--heatmap_mv_var")
        cmd += args.train_extra
        run_step("Step 2/4 train", cmd)

    # ── Step 3: render ──────────────────────────────────
    if args.skip_render:
        print("[skip] Step 3 (render)")
    else:
        cmd = [py, os.path.join(CODE_DIR, "render.py"),
               "-m", model, "--iteration", str(args.iteration)]
        run_step("Step 3/4 render", cmd)

    # ── Step 4: metrics ─────────────────────────────────
    if args.skip_metrics:
        print("[skip] Step 4 (metrics)")
    else:
        cmd = [py, os.path.join(CODE_DIR, "metrics.py"), "-m", model]
        run_step("Step 4/4 metrics", cmd)

    print(f"\n{'='*70}\n[ALL DONE] 총 {(time.time()-t_start)/60:.1f}분  →  {model}\n{'='*70}")


if __name__ == "__main__":
    main()
