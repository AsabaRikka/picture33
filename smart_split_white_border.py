#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能识别图片中的白边/分隔线，并切分成去白边的小图。

适用场景：
- 九宫格/多宫格拼图（白色外边距 + 白色分隔线）
- 也可仅做“整体去白边”（当检测不到网格时）

依赖：
  pip install pillow numpy

示例：
  python smart_split_white_border.py input.png -o out
  python smart_split_white_border.py input.png -o out --trim
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


@dataclass
class Segment:
    start: int  # inclusive
    end: int    # exclusive

    @property
    def size(self) -> int:
        return max(0, self.end - self.start)


def _find_content_segments(
    white_ratio_1d: np.ndarray,
    sep_white_ratio: float,
    min_seg: int,
) -> List[Segment]:
    """
    给定每列(或每行)的“白色占比”，识别非分隔区域(内容块)段。
    分隔线/空白：white_ratio >= sep_white_ratio
    内容块：white_ratio < sep_white_ratio
    """
    is_sep = white_ratio_1d >= sep_white_ratio
    segments: List[Segment] = []
    n = int(white_ratio_1d.shape[0])

    i = 0
    while i < n:
        while i < n and is_sep[i]:
            i += 1
        if i >= n:
            break
        j = i
        while j < n and (not is_sep[j]):
            j += 1
        seg = Segment(i, j)
        if seg.size >= min_seg:
            segments.append(seg)
        i = j
    return segments


def _trim_border_by_whiteness(
    arr_rgb: np.ndarray,
    white_threshold: int,
    trim_white_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    在一个裁剪区域内进一步“仅裁掉连续白边”（不会删掉中间的白色区域）。
    返回 (left, top, right, bottom) 相对于该 arr_rgb 的坐标，right/bottom为exclusive。
    """
    h, w, _ = arr_rgb.shape
    if h == 0 or w == 0:
        return (0, 0, w, h)

    white_mask = np.all(arr_rgb >= white_threshold, axis=2)
    row_white_ratio = white_mask.mean(axis=1)  # (h,)
    col_white_ratio = white_mask.mean(axis=0)  # (w,)

    top = 0
    while top < h and row_white_ratio[top] >= trim_white_ratio:
        top += 1
    bottom = h
    while bottom > top and row_white_ratio[bottom - 1] >= trim_white_ratio:
        bottom -= 1

    left = 0
    while left < w and col_white_ratio[left] >= trim_white_ratio:
        left += 1
    right = w
    while right > left and col_white_ratio[right - 1] >= trim_white_ratio:
        right -= 1

    return (left, top, right, bottom)


def split_image_3x3(
    img: Image.Image,
    *,
    enable_trim: bool = True,
    trim_white_ratio: float = 0.995,
    white_threshold: int = 245,
) -> Tuple[List[Tuple[Tuple[int, int, int, int], Image.Image]], dict]:
    """
    将图片直接切分为 3x3 九等分（等大小切分，不识别白边）。
    可选：对每个切片进行去白边处理。
    返回：[(bbox, tile_img), ...], 以及 metadata
    """
    w, h = img.size
    tile_w = w // 3
    tile_h = h // 3

    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)

    tiles: List[Tuple[Tuple[int, int, int, int], Image.Image]] = []
    meta = {
        "image_size": {"width": int(w), "height": int(h)},
        "params": {
            "enable_trim": bool(enable_trim),
            "trim_white_ratio": float(trim_white_ratio),
            "white_threshold": int(white_threshold),
        },
        "detected_grid": {"cols": 3, "rows": 3},
        "mode": "fixed_3x3",
        "tiles": [],
    }

    for row in range(3):
        for col in range(3):
            left = col * tile_w
            top = row * tile_h
            right = left + tile_w if col < 2 else w
            bottom = top + tile_h if row < 2 else h

            tile = img_rgb.crop((left, top, right, bottom))

            if enable_trim:
                tile_arr = np.array(tile)
                tl, tt, tr, tb = _trim_border_by_whiteness(tile_arr, white_threshold, trim_white_ratio)
                left += tl
                top += tt
                right = left + (tr - tl)
                bottom = top + (tb - tt)
                tile = img_rgb.crop((left, top, right, bottom))

            bbox = (int(left), int(top), int(right), int(bottom))
            tiles.append((bbox, tile))
            meta["tiles"].append({"row": row, "col": col, "bbox": list(bbox)})

    return tiles, meta


def split_image(
    img: Image.Image,
    *,
    white_threshold: int = 245,
    sep_white_ratio: float = 0.985,
    min_seg: int = 32,
    enable_inner_trim: bool = True,
    trim_white_ratio: float = 0.995,
) -> Tuple[List[Tuple[Tuple[int, int, int, int], Image.Image]], dict]:
    """
    返回：[(bbox, tile_img), ...], 以及 metadata
    bbox 为原图坐标 (left, top, right, bottom) right/bottom exclusive
    """
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)  # (H,W,3)
    h, w, _ = arr.shape

    # 判定“白色像素”
    white_mask = np.all(arr >= white_threshold, axis=2)
    col_white_ratio = white_mask.mean(axis=0)
    row_white_ratio = white_mask.mean(axis=1)

    x_segs = _find_content_segments(col_white_ratio, sep_white_ratio, min_seg)
    y_segs = _find_content_segments(row_white_ratio, sep_white_ratio, min_seg)

    tiles: List[Tuple[Tuple[int, int, int, int], Image.Image]] = []
    meta = {
        "image_size": {"width": int(w), "height": int(h)},
        "params": {
            "white_threshold": int(white_threshold),
            "sep_white_ratio": float(sep_white_ratio),
            "min_seg": int(min_seg),
            "enable_inner_trim": bool(enable_inner_trim),
            "trim_white_ratio": float(trim_white_ratio),
        },
        "detected_grid": {"cols": len(x_segs), "rows": len(y_segs)},
        "segments": {
            "x": [{"start": s.start, "end": s.end} for s in x_segs],
            "y": [{"start": s.start, "end": s.end} for s in y_segs],
        },
        "tiles": [],
    }

    # 若检测不到多块内容，则退化为：整体去白边
    if len(x_segs) <= 1 or len(y_segs) <= 1:
        l, t, r, b = _trim_border_by_whiteness(arr, white_threshold, trim_white_ratio)
        bbox = (int(l), int(t), int(r), int(b))
        tile = img_rgb.crop(bbox)
        tiles.append((bbox, tile))
        meta["tiles"].append({"row": 0, "col": 0, "bbox": list(bbox)})
        meta["mode"] = "single_crop"
        return tiles, meta

    meta["mode"] = "grid_split"

    for row_i, ys in enumerate(y_segs):
        for col_i, xs in enumerate(x_segs):
            bbox = (xs.start, ys.start, xs.end, ys.end)
            tile = img_rgb.crop(bbox)
            if enable_inner_trim:
                tile_arr = np.array(tile)
                il, it, ir, ib = _trim_border_by_whiteness(tile_arr, white_threshold, trim_white_ratio)
                # 将内裁剪坐标映射回原图
                bbox2 = (bbox[0] + il, bbox[1] + it, bbox[0] + ir, bbox[1] + ib)
                tile = img_rgb.crop(bbox2)
                bbox = bbox2
            tiles.append((tuple(map(int, bbox)), tile))
            meta["tiles"].append({"row": int(row_i), "col": int(col_i), "bbox": list(map(int, bbox))})

    return tiles, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="智能识别白边并切分图片（九宫格/多宫格）")
    ap.add_argument("input", help="输入图片路径")
    ap.add_argument("-o", "--out", default="out_tiles", help="输出目录（默认：out_tiles）")
    ap.add_argument("--white-threshold", type=int, default=245, help="判定白色像素阈值（RGB都>=此值视为白）")
    ap.add_argument("--sep-ratio", type=float, default=0.985, help="分隔线判定：该行/列白色像素占比阈值")
    ap.add_argument("--min-seg", type=int, default=32, help="最小内容块宽/高（像素），小于此值会被忽略")
    ap.add_argument("--no-trim", action="store_true", help="不进行每个切片的二次去白边")
    ap.add_argument("--trim-ratio", type=float, default=0.995, help="二次去白边：该行/列白色占比阈值")
    ap.add_argument("--prefix", default="tile", help="输出文件名前缀（默认：tile）")
    ap.add_argument("--format", default="png", choices=["png", "jpg", "jpeg", "webp"], help="输出格式")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    img = Image.open(args.input)
    tiles, meta = split_image(
        img,
        white_threshold=args.white_threshold,
        sep_white_ratio=args.sep_ratio,
        min_seg=args.min_seg,
        enable_inner_trim=(not args.no_trim),
        trim_white_ratio=args.trim_ratio,
    )

    # 写出切片
    fmt = args.format.lower()
    if fmt == "jpeg":
        fmt = "jpg"

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    for i, (bbox, tile) in enumerate(tiles):
        # row/col 仅在 grid_split 时有意义；single_crop 时就是 0,0
        tinfo = meta["tiles"][i]
        row = tinfo.get("row", 0)
        col = tinfo.get("col", 0)
        out_name = f"{input_stem}_{i:02d}.{fmt}"
        out_path = os.path.join(args.out, out_name)

        # jpg 不支持透明；这里统一转 RGB
        if fmt in ("jpg", "jpeg"):
            tile = tile.convert("RGB")
            tile.save(out_path, quality=95)
        else:
            tile.save(out_path)

    # 写出元数据
    meta_path = os.path.join(args.out, f"{input_stem}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"完成：输出 {len(tiles)} 张切片到：{args.out}")
    print(f"模式：{meta.get('mode')}，检测到 {meta['detected_grid']['rows']} 行 x {meta['detected_grid']['cols']} 列")
    print(f"元数据：{meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

