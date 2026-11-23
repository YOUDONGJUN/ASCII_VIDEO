"""
code_reel.py - 코드 에디터 + ASCII 영상 + 터미널이 한 화면에 나오는 '코딩 릴스' 영상 만들기

화면 구성 (1920x1080)
  왼쪽   : 코드 에디터. ascii_video.py 코드가 타이핑되며 스크롤됨
  오른쪽 : ASCII로 변환된 영상이 실시간 재생되는 출력 창
  아래   : 터미널. 실행 명령과 프레임 로그, 진행 바

사용법 (ASCII_VIDEO 폴더에서):
  .venv\\Scripts\\python code_reel.py chiikawa\\input\\source.mp4 chiikawa\\output\\code_reel.mp4
  .venv\\Scripts\\python code_reel.py chiikawa\\input\\source.mp4 chiikawa\\output\\test.png --preview 20

옵션:
  --code ascii_video.py   왼쪽 에디터에 타이핑될 코드 파일
  --project chiikawa      화면에 표시할 프로젝트 이름 (기본: 작품 폴더 이름)
  --name usagi.mp4        터미널 명령에 표시할 파일 이름 (기본: <프로젝트>.mp4)
  --hi 1.45               (dark) 이 밝기에서 가장 빽빽한 문자. 어두운 영상은 1.0 정도로 낮추기
  --chroma-dim 0.4        (dark) 흰/크림색 칸을 어둡게 하는 정도. 0이면 끔
  --subs FILE.srt         박힌 자막을 지우고 출력 창에 이 자막을 얹기 (기본: 영상 옆 subs.srt)
  --no-subs               자막 처리 끄기
"""
import argparse
import bisect
import os
import subprocess
import time
from collections import deque
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

import ascii_video as av
import subtitles as st

W, H = 1920, 1080
TITLE_H, TAB_H, STATUS_H = 34, 32, 26
SPLIT_X = 760                       # 코드 창 | 출력 창 경계
CODE_SIZE, CODE_LH, CODE_BASE = 15, 22, 16
GUTTER_W = 58
ANCHOR_LINE = 30                    # 커서가 이 줄 아래로 내려가면 스크롤
A_CELL_W, A_CELL_H, A_COLS = 7, 13, 164   # 출력 창 ASCII 글자 크기/개수
TERM_SIZE, TERM_LH = 14, 20
TYPE_END = 0.9                      # 영상 길이의 90% 지점에서 타이핑 완료

C = dict(
    bg=(30, 30, 30), title=(24, 24, 24), tabbar=(37, 37, 38), border=(52, 52, 52),
    text=(212, 212, 212), dim=(140, 140, 140), lineno=(110, 118, 129), lineno_cur=(198, 198, 198),
    cur_line=(42, 45, 46), cursor=(220, 220, 220), accent=(229, 143, 176), status_text=(48, 24, 36),
    panel=(10, 10, 16), term=(24, 24, 24), green=(106, 190, 120), yellow=(220, 220, 170),
    blue=(86, 156, 214),
)
FONT_DIR = "C:/Windows/Fonts"


def load(name, size):
    path = os.path.join(FONT_DIR, name)
    return ImageFont.truetype(path, size) if os.path.exists(path) else av.load_font(size)


def is_wide(ch):
    o = ord(ch)
    return (0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F or 0xAC00 <= o <= 0xD7AF
            or 0x3000 <= o <= 0x303F or ch in "▶●")


class MixedFont:
    """영문은 고정폭(또는 UI) 폰트, 한글은 맑은 고딕으로 섞어 그리기"""

    def __init__(self, size, bold=False, ui=False):
        lat = ("segoeuib.ttf" if bold else "segoeui.ttf") if ui else ("consolab.ttf" if bold else "consola.ttf")
        self.lat = load(lat, size)
        self.cjk = load("malgunbd.ttf" if bold else "malgun.ttf", size)

    def pick(self, ch):
        return self.cjk if is_wide(ch) else self.lat

    def draw(self, d, x, y_base, text, fill):
        runs = []
        for ch in text:
            f = self.pick(ch)
            if runs and runs[-1][0] is f:
                runs[-1][1] += ch
            else:
                runs.append([f, ch])
        for f, seg in runs:
            d.text((x, y_base), seg, font=f, fill=fill, anchor="ls")
            x += f.getlength(seg)
        return x


def token_color(tt, value):
    if tt in Token.Comment:
        return (106, 153, 85)
    if tt in Token.Literal.String:
        return (206, 145, 120)
    if tt in Token.Literal.Number:
        return (181, 206, 168)
    if tt in Token.Keyword.Constant or tt in Token.Operator.Word:
        return C["blue"]
    if tt in Token.Keyword:
        return C["blue"] if value in ("def", "class", "lambda") else (197, 134, 192)
    if tt in Token.Name.Builtin.Pseudo:
        return C["blue"]
    if tt in Token.Name.Function or tt in Token.Name.Builtin or tt in Token.Name.Decorator:
        return C["yellow"]
    if tt in Token.Name.Class or tt in Token.Name.Namespace:
        return (78, 201, 176)
    if tt in Token.Name:
        return (156, 220, 254)
    return C["text"]


class CodeEditor:
    """코드 전체를 미리 긴 이미지로 그려두고, 프레임마다 잘라서 타이핑 효과를 낸다."""

    def __init__(self, path, width, height):
        self.w, self.h = width, height
        src = open(path, encoding="utf-8").read()
        lines = [[]]
        for tt, val in lex(src, PythonLexer()):
            for i, part in enumerate(val.split("\n")):
                if i:
                    lines.append([])
                if part:
                    lines[-1].append((part, token_color(tt, part)))
        while len(lines) > 1 and not lines[-1]:
            lines.pop()

        mf = MixedFont(CODE_SIZE)
        total_h = len(lines) * CODE_LH + height        # 아래쪽 여백 포함
        normal = Image.new("RGB", (width, total_h), C["bg"])
        hl = Image.new("RGB", (width, total_h), C["bg"])
        dn, dh = ImageDraw.Draw(normal), ImageDraw.Draw(hl)
        self.xs, self.texts = [], []
        for i, segs in enumerate(lines):
            top = i * CODE_LH
            base = top + CODE_BASE
            dh.rectangle([0, top, width, top + CODE_LH - 1], fill=C["cur_line"])
            dn.text((GUTTER_W - 18, base), str(i + 1), font=mf.lat, fill=C["lineno"], anchor="rs")
            dh.text((GUTTER_W - 18, base), str(i + 1), font=mf.lat, fill=C["lineno_cur"], anchor="rs")
            x, xs, text = float(GUTTER_W), [float(GUTTER_W)], ""
            for seg, col in segs:
                for ch in seg:
                    f = mf.pick(ch)
                    if ch != " ":
                        dn.text((x, base), ch, font=f, fill=col, anchor="ls")
                        dh.text((x, base), ch, font=f, fill=col, anchor="ls")
                    x += f.getlength(ch)
                    xs.append(x)
                    text += ch
            self.xs.append(xs)
            self.texts.append(text)
        self.normal, self.hl = np.asarray(normal), np.asarray(hl)

        # 타이핑 순서: 줄 앞 들여쓰기는 자동(가중치 0), 나머지 글자와 줄바꿈은 1
        weights, self.line_start, pos = [], [], 0
        for i, t in enumerate(self.texts):
            self.line_start.append(pos)
            lead = len(t) - len(t.lstrip(" "))
            weights += [0 if j < lead else 1 for j in range(len(t))]
            pos += len(t)
            if i < len(self.texts) - 1:
                weights.append(1)
                pos += 1
        self.cumw = np.cumsum(weights)

    @property
    def n_lines(self):
        return len(self.texts)

    def cursor_at(self, progress):
        k = int(np.searchsorted(self.cumw, progress * self.cumw[-1], side="right"))
        line = bisect.bisect_right(self.line_start, k) - 1
        col = min(k - self.line_start[line], len(self.texts[line]))
        return line, col

    def render(self, progress, cursor_on):
        line, col = self.cursor_at(progress)
        frac = col / max(1, len(self.texts[line]))
        s = int(max(0.0, (line + frac - ANCHOR_LINE) * CODE_LH))
        view = self.normal[s:s + self.h].copy()
        ly = line * CODE_LH - s
        view[ly:ly + CODE_LH] = self.hl[line * CODE_LH:(line + 1) * CODE_LH]
        view[ly + CODE_LH:] = C["bg"]                          # 아직 안 친 줄
        cx = int(round(self.xs[line][col]))
        view[ly:ly + CODE_LH, cx:] = C["cur_line"]             # 아직 안 친 글자
        if cursor_on:
            view[ly + 3:ly + CODE_LH - 3, cx:cx + 2] = C["cursor"]
        return view, line, col


def project_name(input_path):
    """chiikawa\\input\\source.mp4 -> 'chiikawa' (input 폴더의 상위 폴더 이름)"""
    parent = os.path.dirname(os.path.abspath(input_path))
    if os.path.basename(parent).lower() == "input":
        parent = os.path.dirname(parent)
    return os.path.basename(parent)


def build_base(code_name, out_name, project):
    img = Image.new("RGB", (W, H), C["bg"])
    d = ImageDraw.Draw(img)
    ui, ui_b, mono = MixedFont(13, ui=True), MixedFont(13, bold=True, ui=True), MixedFont(13)

    # 제목 표시줄
    d.rectangle([0, 0, W, TITLE_H - 1], fill=C["title"])
    for i, col in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        cx = 22 + i * 22
        d.ellipse([cx - 6, TITLE_H / 2 - 6, cx + 6, TITLE_H / 2 + 6], fill=col)
    title = f"{code_name} — {project}"
    ui.draw(d, W / 2 - ui.lat.getlength(title) / 2, TITLE_H / 2 + 5, title, C["dim"])

    # 탭 표시줄
    y0, y1 = TITLE_H, TITLE_H + TAB_H
    d.rectangle([0, y0, W, y1 - 1], fill=C["tabbar"])
    d.rectangle([0, y0, 190, y1 - 1], fill=C["bg"])
    d.rectangle([0, y0, 190, y0 + 1], fill=C["accent"])
    d.ellipse([14, y0 + 12, 22, y0 + 20], fill=(255, 212, 59))
    ui_b.draw(d, 30, y0 + 21, code_name, C["text"])
    d.rectangle([SPLIT_X + 1, y0, SPLIT_X + 240, y1 - 1], fill=C["panel"])
    d.rectangle([SPLIT_X + 1, y0, SPLIT_X + 240, y0 + 1], fill=C["accent"])
    ui_b.draw(d, SPLIT_X + 16, y0 + 21, f"▶ 출력  {out_name}", C["text"])

    # 창 경계선
    d.rectangle([SPLIT_X, TITLE_H, SPLIT_X, H - STATUS_H], fill=C["border"])
    d.rectangle([0, y1 - 1, W, y1 - 1], fill=C["border"])
    return img, (ui, ui_b, mono)


def main():
    ap = argparse.ArgumentParser(description="코드 에디터 + ASCII 영상 + 터미널 합성 영상 만들기")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--code", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ascii_video.py"))
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--theme", choices=av.THEMES, default="dark")
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--hi", type=float, default=1.45)
    ap.add_argument("--chroma-dim", type=float, default=0.4)
    ap.add_argument("--subs", default=None)
    ap.add_argument("--no-subs", action="store_true")
    args = ap.parse_args()
    srt = None if args.no_subs else (args.subs or st.find_srt(args.input))
    subs = st.Subtitles(srt) if srt else None
    project = args.project or project_name(args.input)
    args.name = args.name or f"{project}.mp4"

    ffmpeg, ffprobe = av.find_tool("ffmpeg"), av.find_tool("ffprobe")
    info = av.probe(ffprobe, args.input)
    fps = float(Fraction(info["fps"]))
    n_total = max(1, int(info["duration"] * fps))
    theme = dict(av.THEMES[args.theme], bg=C["panel"])
    if args.theme == "dark":
        # 작은 출력 창에서는 밝은 배경을 덜 빽빽하게, 윤곽선은 더 밝게 해서 캐릭터가 도드라지게
        theme.update(hi=args.hi, edge_amt=0.75, chroma_dim=args.chroma_dim)

    # 출력 창 ASCII 크기
    cols = A_COLS
    rows = round(cols * info["h"] / info["w"] * A_CELL_W / A_CELL_H)
    aw, ah = cols * A_CELL_W, rows * A_CELL_H
    sw, sh = cols * av.SUB_W, rows * av.SUB_H
    glyphs = av.build_glyphs(A_CELL_W, A_CELL_H)
    ax = SPLIT_X + 1 + (W - SPLIT_X - 1 - aw) // 2
    ay = TITLE_H + TAB_H + 6

    # 터미널 영역
    term_y = ay + ah + 6
    term_x, term_w, term_h = SPLIT_X + 1, W - SPLIT_X - 1, H - STATUS_H - term_y
    code_y, code_h = TITLE_H + TAB_H, H - STATUS_H - TITLE_H - TAB_H

    out_name = os.path.splitext(args.name)[0] + "_ascii.mp4"
    base_img, (ui, ui_b, mono) = build_base(os.path.basename(args.code), out_name, project)
    d = ImageDraw.Draw(base_img)
    d.rectangle([SPLIT_X + 1, ay - 6, W, ay + ah + 5], fill=C["panel"])
    d.rectangle([term_x, term_y, W, H - STATUS_H], fill=C["term"])
    d.rectangle([term_x, term_y, W, term_y], fill=C["border"])
    x = term_x + 18
    for label in ("문제", "출력", "디버그 콘솔", "터미널"):
        active = label == "터미널"
        x2 = ui_b.draw(d, x, term_y + 22, label, C["text"] if active else C["dim"])
        if active:
            d.rectangle([x, term_y + 28, x2, term_y + 29], fill=C["accent"])
        x = x2 + 26
    base = np.asarray(base_img).copy()

    editor = CodeEditor(args.code, SPLIT_X, code_h)
    term_font = MixedFont(TERM_SIZE)
    status_font = MixedFont(13, ui=True)
    cmd = f"python {os.path.basename(args.code)} {args.name} --theme {args.theme}"
    info_line = f"원본 {info['w']}x{info['h']} @ {fps:g}fps -> 글자 {cols}x{rows} -> 출력 {aw}x{ah}"
    logs = deque(maxlen=9)
    text_top = term_y + 40
    duration = info["duration"]

    def compose(frame, i, render_ms, stats):
        t = i / fps
        out = base.copy()
        # 1) 코드 에디터
        progress = min(1.0, t / (duration * TYPE_END))
        cursor_on = progress < 1.0 or int(t * 2) % 2 == 0
        view, line, col = editor.render(progress, cursor_on)
        out[code_y:code_y + code_h, :SPLIT_X] = view
        # 2) ASCII 출력
        out[ay:ay + ah, ax:ax + aw] = frame
        # 3) 터미널
        timg = Image.new("RGB", (term_w, H - STATUS_H - text_top), C["term"])
        td = ImageDraw.Draw(timg)
        y = TERM_LH - 4
        x = term_font.draw(td, 18, y, f"PS C:\\{project}> ", C["green"])
        term_font.draw(td, x, y, cmd, C["yellow"])
        term_font.draw(td, 18, y + TERM_LH, info_line, C["dim"])
        if i % 2 == 0:
            logs.append(f"frame {i + 1:04d}/{n_total}  t={t:05.2f}s  lum={stats.get('lum', 0):.3f}  "
                        f"edges={stats.get('edges', 0):4d}  render={render_ms:4.1f}ms")
        for k, line_txt in enumerate(logs):
            term_font.draw(td, 18, y + TERM_LH * (k + 2), line_txt, C["text"])
        pct = (i + 1) / n_total
        bar_n = 42
        filled = int(pct * bar_n)
        py = y + TERM_LH * (len(logs) + 2) + 4
        x = term_font.draw(td, 18, py, "█" * filled, C["accent"])
        x = term_font.draw(td, x, py, "░" * (bar_n - filled), (80, 80, 80))
        term_font.draw(td, x + 12, py, f"{pct * 100:5.1f}%   {int(t) // 60:02d}:{int(t) % 60:02d} / "
                                       f"{int(duration) // 60:02d}:{int(duration) % 60:02d}", C["text"])
        out[text_top:H - STATUS_H, term_x:] = np.asarray(timg)
        # 4) 상태 표시줄
        simg = Image.new("RGB", (W, STATUS_H), C["accent"])
        sd = ImageDraw.Draw(simg)
        status_font.draw(sd, 14, 18, f"●  실행 중: {os.path.basename(args.code)}", C["status_text"])
        right = f"줄 {line + 1}, 열 {col + 1}     공백: 4     UTF-8     Python 3.14"
        status_font.draw(sd, W - 14 - status_font.lat.getlength(right) - 40, 18, right, C["status_text"])
        out[H - STATUS_H:] = np.asarray(simg)
        return out

    start = args.preview or 0.0
    dec_cmd = [ffmpeg, "-v", "error", "-ss", str(start), "-i", args.input]
    if args.preview is not None:
        dec_cmd += ["-frames:v", "1"]
    if subs:  # 박힌 자막을 지우려면 원본 해상도로 받아서 직접 줄인다
        fw, fh = info["w"], info["h"]
    else:
        fw, fh = sw, sh
        dec_cmd += ["-vf", f"scale={sw}:{sh}:flags=area"]
    dec_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    frame_bytes = fw * fh * 3
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if subs:
        print(f"자막: {srt} ({len(subs.cues)}줄)")

    def next_frame(t):
        raw = dec.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            return None
        small = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3)
        if subs:
            small = subs.prepare(small, t, (sw, sh))
        stats = {}
        t0 = time.perf_counter()
        ascii_img = av.render_frame(small, rows, cols, glyphs, theme, True, 0.9, stats)
        ms = (time.perf_counter() - t0) * 1000
        if subs:
            ascii_img = subs.draw(ascii_img, t, args.theme)
        return ascii_img, ms, stats

    if args.preview is not None:
        i = int(start * fps)
        ascii_img, ms, stats = next_frame(start + 0.5 / fps)
        dec.wait()
        for j in range(i - 16, i, 2):               # 미리보기용 로그 채우기
            if j >= 0:
                compose(ascii_img, j, ms, stats)
        Image.fromarray(compose(ascii_img, i, ms, stats)).save(args.output)
        print("저장:", args.output)
        return

    enc_cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", info["fps"], "-i", "-"]
    if info["has_audio"]:
        enc_cmd += ["-i", args.input, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    enc_cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-shortest", "-movflags", "+faststart", args.output]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=10 ** 8)

    print(f"코드 {editor.n_lines}줄 타이핑 + ASCII {cols}x{rows} 합성 -> {args.output}")
    t0, i = time.time(), 0
    while True:
        res = next_frame((i + 0.5) / fps)
        if res is None:
            break
        enc.stdin.write(compose(res[0], i, res[1], res[2]).tobytes())
        i += 1
        if i % 30 == 0:
            el = time.time() - t0
            print(f"\r{i}/{n_total} 프레임  {i / el:.1f} fps  남은 시간 ~{(n_total - i) / (i / el):.0f}초   ",
                  end="", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"\n완료: {args.output}  ({i} 프레임, {time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
