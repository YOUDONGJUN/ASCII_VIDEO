"""
subtitles.py - 영상에 박혀 있는 자막을 지우고, SRT 자막을 깨끗한 한글로 다시 얹기

ASCII로 바꾸면 박힌 자막(흰 글자 + 검은 테두리)은 몇 칸짜리 기호 뭉치가 되어 읽을 수 없다.
그래서
  1) 변환 전: 원본 해상도 프레임에서 자막 픽셀을 찾아 주변 색으로 메운다 (자막이 나오는 시간에만)
  2) 변환 후: ASCII 화면 위에 SRT 자막을 한글 폰트로 그린다

<작품>/input/subs.srt 가 있으면 ascii_video.py / code_reel.py 가 자동으로 사용한다. (--no-subs 로 끄기)
"""
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BAND = (0.80, 0.985)     # 박힌 자막을 찾을 세로 범위 (프레임 높이 비율)
WHITE, DARK = 200, 90    # 자막 글자(흰색) / 테두리(검은색) 판정 기준 (0~255)
PAD = 2 / 30             # 박힌 자막 지우기를 앞뒤로 이만큼(초) 더 적용

# 테마별 자막 색: (글자, 테두리)
STYLES = {
    "dark": ((255, 250, 240), (10, 10, 16)),
    "paper": ((44, 30, 26), (248, 244, 235)),
}

FONTS = ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
         "/System/Library/Fonts/AppleSDGothicNeo.ttc",
         "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]

TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(path):
    cues = []
    for block in re.split(r"\n\s*\n", open(path, encoding="utf-8-sig").read().strip()):
        lines = block.strip().splitlines()
        for k, line in enumerate(lines):
            m = TIME.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                cues.append((a, b, "\n".join(lines[k + 1:]).strip()))
                break
    return cues


def find_srt(input_path):
    """영상과 같은 폴더의 subs.srt"""
    path = os.path.join(os.path.dirname(os.path.abspath(input_path)), "subs.srt")
    return path if os.path.exists(path) else None


def dilate(m, r):
    """r 픽셀 반경(정사각형)으로 True 영역 넓히기"""
    out = m.copy()
    for d in range(1, r + 1):
        out[:, d:] |= m[:, :-d]
        out[:, :-d] |= m[:, d:]
    row = out.copy()
    for d in range(1, r + 1):
        out[d:] |= row[:-d]
        out[:-d] |= row[d:]
    return out


def box_sum(a, r):
    """(2r+1)x(2r+1) 상자 안의 합 (가장자리는 0으로 채움)"""
    pad = ((r + 1, r), (r + 1, r)) + ((0, 0),) * (a.ndim - 2)
    c = np.pad(a, pad).cumsum(0).cumsum(1)
    k = 2 * r + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def grow_run(idx, start, gap):
    """정렬된 인덱스 idx에서 start 위치부터 간격이 gap 미만인 동안 양쪽으로 넓힌 (처음, 끝)"""
    lo = hi = start
    while lo > 0 and idx[lo] - idx[lo - 1] < gap:
        lo -= 1
    while hi < len(idx) - 1 and idx[hi + 1] - idx[hi] < gap:
        hi += 1
    return idx[lo], idx[hi] + 1


def erase_burned(frame):
    """흰 글자 + 검은 테두리 자막을 찾아 주변 색으로 메운 프레임을 돌려준다."""
    h, w, _ = frame.shape
    s = h / 720
    r = max(1, round(4 * s))
    y0, y1 = int(h * BAND[0]), int(h * BAND[1])
    band = frame[y0:y1].astype(np.float32)
    white = band.min(axis=2) > WHITE
    dark = band.max(axis=2) < DARK
    # 흰 글자(테두리 옆) + 검은 테두리(흰 글자 옆)
    seed = (white & dilate(dark, r)) | (dark & dilate(white, r))

    # 화면 가운데에서 시작해 글자 사이 빈 칸이 좁은 동안만 자막으로 본다 (양옆 배경 제외)
    xs = np.nonzero(seed.any(axis=0))[0]
    if len(xs) == 0:
        return frame
    i = int(np.argmin(np.abs(xs - w / 2)))
    if abs(xs[i] - w / 2) > 0.25 * w:
        return frame
    xa, xb = grow_run(xs, i, 0.06 * w)
    seed[:, :xa] = False
    seed[:, xb:] = False
    # 세로로는 seed가 가장 많은 줄에서 시작해 이어진 줄들만
    prof = seed.sum(axis=1)
    ys = np.nonzero(prof > 0.02 * (xb - xa))[0]
    if len(ys) == 0:
        return frame
    ya, yb = grow_run(ys, int(np.searchsorted(ys, prof.argmax())), max(2, round(4 * s)))
    seed[:ya] = False
    seed[yb:] = False

    # 글자 + 테두리를 구멍으로 보고 (닫힘 연산으로 굵은 글자 속까지 포함)
    hole = dilate(seed, max(2, round(7 * s)))
    k = max(2, round(8 * s))
    hole = ~dilate(~dilate(hole, k), k)
    # 바깥 색을 평균내어 안쪽으로 채워 나간 뒤, 채운 부분을 한 번 더 부드럽게
    known = ~hole
    out = band
    for r in (8, 20, 48):
        r = max(2, round(r * s))
        wsum = box_sum(known.astype(np.float32), r)
        fill = box_sum(out * known[..., None], r) / np.maximum(wsum, 1e-6)[..., None]
        new = ~known & (wsum > 0.5)
        out[new] = fill[new]
        known |= new
    r = max(2, round(6 * s))
    soft = box_sum(out, r) / box_sum(np.ones(out.shape[:2], np.float32), r)[..., None]
    out[hole] = soft[hole]
    frame = frame.copy()
    frame[y0:y1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return frame


def load_font(size):
    for path in FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


class Subtitles:
    def __init__(self, path):
        self.path = path
        self.cues = parse_srt(path)
        self.cache = {}

    def text_at(self, t, pad=0.0):
        for a, b, text in self.cues:
            if a - pad <= t < b + pad:
                return text
        return None

    def prepare(self, frame, t, size):
        """원본 해상도 프레임 -> 박힌 자막 지우기 -> ASCII 샘플링 크기로 줄이기"""
        if self.text_at(t, PAD) is not None:
            frame = erase_burned(frame)
        return np.asarray(Image.fromarray(frame).resize(size, Image.BOX))

    def sprite(self, text, height, style):
        key = (text, height, style)
        if key not in self.cache:
            fill, stroke = STYLES[style]
            size = max(12, round(height * 0.058))
            sw = max(2, round(size * 0.13))
            font = load_font(size)
            d = ImageDraw.Draw(Image.new("L", (1, 1)))
            box = d.multiline_textbbox((0, 0), text, font=font, stroke_width=sw, align="center", spacing=size // 4)
            img = Image.new("RGBA", (box[2] - box[0] + 4, box[3] - box[1] + 4), stroke + (0,))
            ImageDraw.Draw(img).multiline_text((2 - box[0], 2 - box[1]), text, font=font, fill=fill + (255,),
                                               stroke_width=sw, stroke_fill=stroke + (255,),
                                               align="center", spacing=size // 4)
            self.cache[key] = np.asarray(img, dtype=np.float32) / 255.0
        return self.cache[key]

    def draw(self, img, t, style="dark"):
        """ASCII 화면(img) 아래쪽 가운데에 자막을 그린다."""
        text = self.text_at(t)
        if not text:
            return img
        h, w, _ = img.shape
        sp = self.sprite(text, h, style)
        sh, sw = sp.shape[:2]
        x = (w - sw) // 2
        y = round(h * 0.94) - sh
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + sw, w), min(y + sh, h)
        part = sp[y0 - y:y1 - y, x0 - x:x1 - x]
        a = part[..., 3:]
        img = img.copy()
        region = img[y0:y1, x0:x1].astype(np.float32) / 255.0
        img[y0:y1, x0:x1] = ((region * (1 - a) + part[..., :3] * a) * 255 + 0.5).astype(np.uint8)
        return img
