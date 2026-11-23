"""
ascii_video.py - 영상을 컬러 ASCII 문자 영상으로 변환

사용법 (ASCII_VIDEO 폴더에서):
  .venv\\Scripts\\python ascii_video.py chiikawa\\input\\source.mp4 chiikawa\\output\\ascii.mp4
  .venv\\Scripts\\python ascii_video.py chiikawa\\input\\source.mp4 chiikawa\\output\\test.png --preview 12.5

옵션:
  --cols 240            가로 글자 수 (클수록 정밀, 출력 폭 = cols x 8px)
  --theme dark|paper    dark = 검은 배경(코드 화면 느낌), paper = 밝은 종이 배경
  --no-edges            윤곽선을 방향 문자( | / - \\ )로 그리는 기능 끄기
  --edge-thr 0.9        윤곽선으로 판정할 기준 (낮을수록 윤곽선 문자가 많아짐)
  --start / --duration  일부 구간만 변환 (초)
  --preview T           T초 지점 한 프레임만 PNG로 저장 (출력 경로를 .png로)
  --subs FILE.srt       박힌 자막을 지우고 이 자막을 깨끗한 한글로 얹기 (기본: 영상 옆 subs.srt)
  --no-subs             자막 처리 끄기
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import subtitles as st

RAMP = " .:-=+*#%@"          # 밝기(또는 어두움) 순서 문자
EDGE_CHARS = "|/-\\"         # 윤곽선 방향 문자 (세로, /, 가로, \)
CELL_W, CELL_H = 8, 15       # 출력 글자 한 칸 크기(px)
SUB_W, SUB_H = 4, 8          # 원본을 글자 한 칸당 이만큼의 픽셀로 샘플링

THEMES = {
    # invert=False: 밝을수록 빽빽한 문자 / True: 어두울수록 빽빽한 문자
    "dark": dict(bg=(10, 10, 16), invert=False, lo=0.10, hi=1.0, gamma=1.0,
                 sat=1.6, gain=1.12, edge_tint=(255, 246, 232), edge_amt=0.55),
    "paper": dict(bg=(248, 244, 235), invert=True, lo=0.03, hi=0.55, gamma=0.6,
                  sat=1.7, gain=0.80, edge_tint=(58, 40, 34), edge_amt=0.55),
}

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def find_tool(name):
    path = shutil.which(name)
    if path:
        return path
    # winget으로 설치했지만 PATH가 아직 반영 안 된 경우
    pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet",
                           "Packages", "Gyan.FFmpeg*", "*", "bin", name + ".exe")
    hits = glob.glob(pattern)
    if hits:
        return hits[0]
    sys.exit(f"{name}을(를) 찾을 수 없습니다. 'winget install Gyan.FFmpeg'로 설치하세요.")


def probe(ffprobe, path):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    return dict(w=int(v["width"]), h=int(v["height"]), fps=v["r_frame_rate"],
                duration=float(info["format"].get("duration", 0)), has_audio=has_audio)


def load_font(size):
    candidates = [
        "C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/consola.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default(size)


def build_glyphs(cell_w=CELL_W, cell_h=CELL_H):
    """문자마다 cell_h x cell_w 크기의 0~1 마스크를 만든다."""
    font = load_font(cell_h - 1)
    masks = []
    for ch in RAMP + EDGE_CHARS:
        img = Image.new("L", (cell_w, cell_h), 0)
        ImageDraw.Draw(img).text((cell_w / 2, cell_h / 2), ch, font=font, fill=255, anchor="mm")
        masks.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(masks)  # (문자 수, CELL_H, CELL_W)


def sobel(gray):
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx, gy


def cell_mean(a, rows, cols):
    return a.reshape(rows, SUB_H, cols, SUB_W, *a.shape[2:]).mean(axis=(1, 3))


def render_frame(frame, rows, cols, glyphs, theme, edges, edge_thr, stats=None):
    _, cell_h, cell_w = glyphs.shape
    img = frame.astype(np.float32) / 255.0                  # (rows*SUB_H, cols*SUB_W, 3)
    color = cell_mean(img, rows, cols)                       # 칸별 평균 색
    lum = color @ LUMA

    # 1) 밝기 -> 문자
    t = 1.0 - lum if theme["invert"] else lum
    t = np.clip((t - theme["lo"]) / (theme["hi"] - theme["lo"]), 0, 1) ** theme["gamma"]
    idx = np.rint(t * (len(RAMP) - 1)).astype(np.int32)

    # 2) 글자 색: 채도/밝기 보정
    g = lum[..., None]
    fg = np.clip((g + (color - g) * theme["sat"]) * theme["gain"], 0, 1)
    if theme.get("chroma_dim"):
        # 채도 낮은(흰/크림색) 칸은 어둡게 -> 색 있는 캐릭터가 도드라짐
        chroma = color.max(axis=2) - color.min(axis=2)
        fg = fg * (1 - theme["chroma_dim"] * (1 - np.clip(chroma / 0.25, 0, 1)))[..., None]

    # 3) 윤곽선 -> 방향 문자
    if edges:
        gray = img @ LUMA
        gx, gy = sobel(gray)
        jxx = cell_mean(gx * gx, rows, cols)
        jyy = cell_mean(gy * gy, rows, cols)
        jxy = cell_mean(gx * gy, rows, cols)
        mag = np.sqrt(jxx + jyy)
        coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
        is_edge = (mag > edge_thr) & (coherence > 0.35)
        # 그라디언트 각도 phi (-90~90도). 윤곽선은 그에 수직
        phi = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy))
        a = np.abs(phi)
        edge_idx = np.where(a < 22.5, 0, np.where(a > 67.5, 2, np.where(phi > 0, 1, 3)))
        idx = np.where(is_edge, len(RAMP) + edge_idx, idx)
        tint = np.array(theme["edge_tint"], dtype=np.float32) / 255.0
        k = theme["edge_amt"]
        fg = np.where(is_edge[..., None], fg * (1 - k) + tint * k, fg)
        if stats is not None:
            stats["edges"] = int(is_edge.sum())
    if stats is not None:
        stats["lum"] = float(lum.mean())

    # 4) 문자 마스크 합성
    bg = np.array(theme["bg"], dtype=np.float32) / 255.0
    mask = glyphs[idx][..., None]                             # (rows, cols, CH, CW, 1)
    out = bg + mask * (fg[:, :, None, None, :] - bg)
    out = out.transpose(0, 2, 1, 3, 4).reshape(rows * cell_h, cols * cell_w, 3)
    return (out * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="영상을 컬러 ASCII 영상으로 변환")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--cols", type=int, default=240)
    ap.add_argument("--theme", choices=THEMES, default="dark")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--edge-thr", type=float, default=0.9)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--subs", default=None)
    ap.add_argument("--no-subs", action="store_true")
    args = ap.parse_args()

    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    info = probe(ffprobe, args.input)
    fps = float(Fraction(info["fps"]))  # "30/1" 같은 분수
    srt = None if args.no_subs else (args.subs or st.find_srt(args.input))
    subs = st.Subtitles(srt) if srt else None
    cols = args.cols
    rows = max(1, round(cols * info["h"] / info["w"] * CELL_W / CELL_H))
    sw, sh = cols * SUB_W, rows * SUB_H
    ow, oh = cols * CELL_W, rows * CELL_H
    oh -= oh % 2  # yuv420p는 짝수 높이 필요
    theme = THEMES[args.theme]
    glyphs = build_glyphs()
    edges = not args.no_edges

    start = args.preview if args.preview is not None else args.start
    dec_cmd = [ffmpeg, "-v", "error", "-ss", str(start), "-i", args.input]
    if args.preview is not None:
        dec_cmd += ["-frames:v", "1"]
    elif args.duration:
        dec_cmd += ["-t", str(args.duration)]
    if subs:  # 박힌 자막을 지우려면 원본 해상도로 받아서 직접 줄인다
        fw, fh = info["w"], info["h"]
    else:
        fw, fh = sw, sh
        dec_cmd += ["-vf", f"scale={sw}:{sh}:flags=area"]
    dec_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"원본 {info['w']}x{info['h']} @ {info['fps']} -> 글자 {cols}x{rows} -> 출력 {ow}x{oh} ({args.theme})")
    if subs:
        print(f"자막: {srt} ({len(subs.cues)}줄)")

    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    frame_bytes = fw * fh * 3

    def convert(raw, t):
        frame = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3)
        if subs:
            frame = subs.prepare(frame, t, (sw, sh))
        out = render_frame(frame, rows, cols, glyphs, theme, edges, args.edge_thr)[:oh]
        return subs.draw(out, t, args.theme) if subs else out

    if args.preview is not None:
        raw = dec.stdout.read(frame_bytes)
        dec.wait()
        Image.fromarray(convert(raw, start + 0.5 / fps)).save(args.output)
        print("저장:", args.output)
        return

    enc_cmd = [ffmpeg, "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-r", info["fps"], "-i", "-"]
    if info["has_audio"]:
        enc_cmd += ["-ss", str(start)]
        if args.duration:
            enc_cmd += ["-t", str(args.duration)]
        enc_cmd += ["-i", args.input, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    enc_cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                "-shortest", "-movflags", "+faststart", args.output]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=10 ** 8)

    total = args.duration or (info["duration"] - start)
    n_total = max(1, int(total * fps))
    t0, n = time.time(), 0
    while True:
        raw = dec.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        enc.stdin.write(convert(raw, start + (n + 0.5) / fps).tobytes())
        n += 1
        if n % 30 == 0:
            el = time.time() - t0
            print(f"\r{n}/{n_total} 프레임  {n / el:.1f} fps  남은 시간 ~{(n_total - n) / (n / el):.0f}초   ",
                  end="", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"\n완료: {args.output}  ({n} 프레임, {time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
