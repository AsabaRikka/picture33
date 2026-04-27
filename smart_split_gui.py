#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Split GUI
图形界面：批量识别白边/分隔线并切分（九宫格/多宫格），输出无白边切片。

特性：
1) 图形界面（Tkinter）
2) 可选择输入文件夹、输出文件夹
3) 批量处理（可选递归子目录），进度条 + 日志

依赖：
  pip install pillow numpy

运行：
  python smart_split_gui.py
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from pathlib import Path
from typing import List, Optional, Tuple
import json

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from tkinter import ttk
except ModuleNotFoundError:  # 某些精简 Python 环境可能不带 tkinter
    tk = None  # type: ignore
    filedialog = None  # type: ignore
    messagebox = None  # type: ignore
    ttk = None  # type: ignore

from PIL import Image

# 复用切分算法（来自同目录脚本）
from smart_split_white_border import split_image, split_image_3x3  # type: ignore


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _last_config_path() -> Path:
    # Store next to this script for portability (no admin rights needed)
    return Path(__file__).resolve().with_name(".smart_split_gui_last.json")


def _load_last_config() -> dict:
    p = _last_config_path()
    try:
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_last_config(data: dict) -> None:
    p = _last_config_path()
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        # Best-effort; GUI should still run even if saving fails
        pass


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(paths)


def _save_tiles_for_one_image(
    img_path: Path,
    out_root: Path,
    *,
    recursive: bool,
    keep_structure: bool,
    white_threshold: int,
    sep_ratio: float,
    min_seg: int,
    trim_ratio: float,
    disable_trim: bool,
    quality_jpg: int,
) -> Tuple[int, Path]:
    """
    输出结构（用户选择：每张图一个子文件夹（推荐））：
      out_root/<stem>/tile_rX_cY_XX.<ext> + tile_meta.json

    若 keep_structure=True 且 recursive=True，则会输出：
      out_root/<relative_dir>/<stem>/...
    """
    rel_parent = img_path.parent
    if keep_structure:
        # 保持输入结构：以输入根目录为基准，在外层调用处会传入 img_path 相对路径
        # 这里由外部决定 out_root 的具体子路径；保持该函数只输出到 out_root/<stem>/
        pass

    ext = img_path.suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    out_dir = out_root / img_path.stem
    _safe_mkdir(out_dir)

    with Image.open(img_path) as img:
        tiles, meta = split_image(
            img,
            white_threshold=white_threshold,
            sep_white_ratio=sep_ratio,
            min_seg=min_seg,
            enable_inner_trim=(not disable_trim),
            trim_white_ratio=trim_ratio,
        )

        # 写切片
        for i, (bbox, tile) in enumerate(tiles):
            tinfo = meta["tiles"][i]
            row = tinfo.get("row", 0)
            col = tinfo.get("col", 0)
            out_name = f"{img_path.stem}_{i:02d}{ext}"
            out_path = out_dir / out_name
            if ext == ".jpg":
                tile = tile.convert("RGB")
                tile.save(out_path, quality=int(quality_jpg))
            else:
                tile.save(out_path)

        # 写元数据
        import json

        meta_path = out_dir / "tile_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return len(tiles), out_dir


def _save_tiles_for_one_image_3x3(
    img_path: Path,
    out_root: Path,
    *,
    disable_trim: bool,
    trim_ratio: float,
    quality_jpg: int,
) -> Tuple[int, Path]:
    """
    3x3 固定切分模式：将图片等分为 9 块
    """
    ext = img_path.suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    out_dir = out_root / img_path.stem
    _safe_mkdir(out_dir)

    with Image.open(img_path) as img:
        tiles, meta = split_image_3x3(
            img,
            enable_trim=(not disable_trim),
            trim_white_ratio=trim_ratio,
            white_threshold=245,
        )

        # 写切片
        for i, (bbox, tile) in enumerate(tiles):
            tinfo = meta["tiles"][i]
            row = tinfo.get("row", 0)
            col = tinfo.get("col", 0)
            out_name = f"{img_path.stem}_{i:02d}{ext}"
            out_path = out_dir / out_name
            if ext == ".jpg":
                tile = tile.convert("RGB")
                tile.save(out_path, quality=int(quality_jpg))
            else:
                tile.save(out_path)

        # 写元数据
        import json

        meta_path = out_dir / "tile_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return len(tiles), out_dir


if tk is not None:
    class App(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("智能去白边切分工具（批量）")
            self.geometry("780x540")

            self._worker: Optional[threading.Thread] = None
            self._stop_event = threading.Event()
            self._msg_q: "queue.Queue[tuple]" = queue.Queue()

            self._build_ui()
            self._restore_last_inputs()
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            self.after(100, self._drain_messages)

        def _restore_last_inputs(self) -> None:
            cfg = _load_last_config()
            # Paths
            in_dir = cfg.get("in_dir")
            out_dir = cfg.get("out_dir")
            if isinstance(in_dir, str):
                self.in_var.set(in_dir)
            if isinstance(out_dir, str):
                self.out_var.set(out_dir)

            # Toggles
            for key, var in (
                ("recursive", self.recursive_var),
                ("keep_structure", self.keep_structure_var),
                ("disable_trim", self.disable_trim_var),
            ):
                v = cfg.get(key)
                if isinstance(v, bool):
                    var.set(v)

            # Split mode
            split_mode = cfg.get("split_mode")
            if isinstance(split_mode, str) and split_mode in ("auto", "3x3"):
                self.split_mode_var.set(split_mode)

            # Params
            params = cfg.get("params")
            if not isinstance(params, dict):
                params = {}

            def _maybe_set_int(var, k):
                v = params.get(k)
                if isinstance(v, int):
                    var.set(v)

            def _maybe_set_float(var, k):
                v = params.get(k)
                if isinstance(v, (int, float)):
                    var.set(float(v))

            _maybe_set_int(self.white_threshold_var, "white_threshold")
            _maybe_set_float(self.sep_ratio_var, "sep_ratio")
            _maybe_set_int(self.min_seg_var, "min_seg")
            _maybe_set_float(self.trim_ratio_var, "trim_ratio")
            _maybe_set_int(self.jpg_quality_var, "jpg_quality")

        def _persist_current_inputs(self) -> None:
            data = {
                "in_dir": self.in_var.get().strip(),
                "out_dir": self.out_var.get().strip(),
                "recursive": bool(self.recursive_var.get()),
                "keep_structure": bool(self.keep_structure_var.get()),
                "disable_trim": bool(self.disable_trim_var.get()),
                "split_mode": self.split_mode_var.get(),
                "params": {
                    "white_threshold": int(self.white_threshold_var.get()),
                    "sep_ratio": float(self.sep_ratio_var.get()),
                    "min_seg": int(self.min_seg_var.get()),
                    "trim_ratio": float(self.trim_ratio_var.get()),
                    "jpg_quality": int(self.jpg_quality_var.get()),
                },
            }
            _save_last_config(data)

        def _on_close(self) -> None:
            # Save current UI values even if not running
            try:
                self._persist_current_inputs()
            finally:
                self.destroy()

        def _build_ui(self) -> None:
            pad = {"padx": 10, "pady": 6}

            frm = ttk.Frame(self)
            frm.pack(fill="both", expand=True)

            # 输入/输出路径
            io = ttk.LabelFrame(frm, text="路径")
            io.pack(fill="x", **pad)

            self.in_var = tk.StringVar()
            self.out_var = tk.StringVar()

            r0 = ttk.Frame(io)
            r0.pack(fill="x", padx=10, pady=6)
            ttk.Label(r0, text="输入文件夹：").pack(side="left")
            ttk.Entry(r0, textvariable=self.in_var).pack(side="left", fill="x", expand=True, padx=6)
            ttk.Button(r0, text="选择…", command=self._choose_in).pack(side="left")

            r1 = ttk.Frame(io)
            r1.pack(fill="x", padx=10, pady=6)
            ttk.Label(r1, text="输出文件夹：").pack(side="left")
            ttk.Entry(r1, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
            ttk.Button(r1, text="选择…", command=self._choose_out).pack(side="left")

            # 选项
            opt = ttk.LabelFrame(frm, text="选项")
            opt.pack(fill="x", **pad)

            self.recursive_var = tk.BooleanVar(value=False)
            self.keep_structure_var = tk.BooleanVar(value=False)
            self.disable_trim_var = tk.BooleanVar(value=False)

            # 切分模式选择
            row_mode = ttk.Frame(opt)
            row_mode.pack(fill="x", padx=10, pady=4)
            self.split_mode_var = tk.StringVar(value="auto")  # "auto"=智能识别, "3x3"=3x3固定切分
            
            ttk.Label(row_mode, text="切分模式：").pack(side="left")
            ttk.Radiobutton(row_mode, text="智能识别（自动检测白边/分隔线）", variable=self.split_mode_var, value="auto").pack(side="left", padx=6)
            ttk.Radiobutton(row_mode, text="3x3固定切分（等大小九宫格）", variable=self.split_mode_var, value="3x3").pack(side="left", padx=6)

            rowa = ttk.Frame(opt)
            rowa.pack(fill="x", padx=10, pady=4)
            ttk.Checkbutton(rowa, text="递归处理子文件夹", variable=self.recursive_var).pack(side="left")
            ttk.Checkbutton(rowa, text="保持输入目录结构（需开启递归）", variable=self.keep_structure_var).pack(side="left", padx=16)
            ttk.Checkbutton(rowa, text="不做每张切片的二次去白边", variable=self.disable_trim_var).pack(side="left", padx=16)

            # 参数（基础）
            rowb = ttk.Frame(opt)
            rowb.pack(fill="x", padx=10, pady=6)

            self.white_threshold_var = tk.IntVar(value=245)
            self.sep_ratio_var = tk.DoubleVar(value=0.985)
            self.min_seg_var = tk.IntVar(value=32)
            self.trim_ratio_var = tk.DoubleVar(value=0.995)
            self.jpg_quality_var = tk.IntVar(value=95)

            def _add_labeled(entry_parent, label, var, width=8):
                ttk.Label(entry_parent, text=label).pack(side="left", padx=(0, 4))
                ttk.Entry(entry_parent, textvariable=var, width=width).pack(side="left", padx=(0, 14))

            _add_labeled(rowb, "白色阈值", self.white_threshold_var, 8)
            _add_labeled(rowb, "分隔线阈值", self.sep_ratio_var, 8)
            _add_labeled(rowb, "最小块像素", self.min_seg_var, 8)
            _add_labeled(rowb, "二次去白边阈值", self.trim_ratio_var, 8)
            _add_labeled(rowb, "JPG质量", self.jpg_quality_var, 6)

            hint = ttk.Label(
                opt,
                text="提示：如果切分不准，可尝试把“白色阈值”调低(如 235) 或把“分隔线阈值”调低(如 0.97)。",
            )
            hint.pack(anchor="w", padx=10, pady=(0, 8))

            # 控制区
            ctrl = ttk.Frame(frm)
            ctrl.pack(fill="x", **pad)

            self.start_btn = ttk.Button(ctrl, text="开始批量处理", command=self._start)
            self.start_btn.pack(side="left")
            self.stop_btn = ttk.Button(ctrl, text="停止", command=self._stop, state="disabled")
            self.stop_btn.pack(side="left", padx=10)

            self.progress = ttk.Progressbar(ctrl, orient="horizontal", length=380, mode="determinate")
            self.progress.pack(side="left", fill="x", expand=True, padx=10)
            self.progress_label = ttk.Label(ctrl, text="0/0")
            self.progress_label.pack(side="left")

            # 日志
            logf = ttk.LabelFrame(frm, text="日志")
            logf.pack(fill="both", expand=True, **pad)

            self.log = tk.Text(logf, height=12, wrap="word")
            self.log.pack(side="left", fill="both", expand=True)
            scr = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
            scr.pack(side="right", fill="y")
            self.log.configure(yscrollcommand=scr.set)

        def _choose_in(self) -> None:
            p = filedialog.askdirectory(title="选择输入文件夹")
            if p:
                self.in_var.set(p)

        def _choose_out(self) -> None:
            p = filedialog.askdirectory(title="选择输出文件夹")
            if p:
                self.out_var.set(p)

        def _append_log(self, s: str) -> None:
            self.log.insert("end", s + "\n")
            self.log.see("end")

        def _set_running(self, running: bool) -> None:
            self.start_btn.configure(state=("disabled" if running else "normal"))
            self.stop_btn.configure(state=("normal" if running else "disabled"))

        def _stop(self) -> None:
            self._stop_event.set()
            self._append_log("收到停止请求：将尽快停止…")

        def _start(self) -> None:
            in_dir = Path(self.in_var.get().strip())
            out_dir = Path(self.out_var.get().strip())

            if not in_dir.exists() or not in_dir.is_dir():
                messagebox.showerror("错误", "请输入有效的输入文件夹。")
                return
            if not out_dir.exists() or not out_dir.is_dir():
                messagebox.showerror("错误", "请输入有效的输出文件夹。")
                return

            # Record last-used parameters as soon as a valid run is initiated
            self._persist_current_inputs()

            if self._worker and self._worker.is_alive():
                messagebox.showwarning("提示", "任务正在运行中。")
                return

            self._stop_event.clear()
            self._set_running(True)
            self.progress["value"] = 0
            self.progress_label.configure(text="0/0")

            # 收集任务
            recursive = bool(self.recursive_var.get())
            paths = _iter_images(in_dir, recursive=recursive)
            if not paths:
                self._set_running(False)
                messagebox.showinfo("提示", "输入文件夹中未找到可处理的图片（png/jpg/webp/bmp/tiff）。")
                return

            self.progress["maximum"] = len(paths)
            self._append_log(f"准备处理 {len(paths)} 张图片…")

            # 启动线程
            self._worker = threading.Thread(
                target=self._run_batch,
                args=(in_dir, out_dir, paths),
                daemon=True,
            )
            self._worker.start()

        def _run_batch(self, in_dir: Path, out_dir: Path, paths: List[Path]) -> None:
            try:
                recursive = bool(self.recursive_var.get())
                keep_structure = bool(self.keep_structure_var.get()) and recursive
                white_threshold = int(self.white_threshold_var.get())
                sep_ratio = float(self.sep_ratio_var.get())
                min_seg = int(self.min_seg_var.get())
                trim_ratio = float(self.trim_ratio_var.get())
                disable_trim = bool(self.disable_trim_var.get())
                quality_jpg = int(self.jpg_quality_var.get())
                split_mode = self.split_mode_var.get()  # "auto" 或 "3x3"

                ok = 0
                fail = 0

                for idx, img_path in enumerate(paths, start=1):
                    if self._stop_event.is_set():
                        self._msg_q.put(("log", "已停止。"))
                        break

                    try:
                        # 输出：每张图一个子文件夹；可选保持输入目录结构
                        if keep_structure:
                            rel = img_path.relative_to(in_dir)
                            out_root = out_dir / rel.parent
                        else:
                            out_root = out_dir

                        if split_mode == "3x3":
                            # 3x3 固定切分模式
                            tiles_count, out_folder = _save_tiles_for_one_image_3x3(
                                img_path,
                                out_root,
                                disable_trim=disable_trim,
                                trim_ratio=trim_ratio,
                                quality_jpg=quality_jpg,
                            )
                        else:
                            # 智能识别模式（默认）
                            tiles_count, out_folder = _save_tiles_for_one_image(
                                img_path,
                                out_root,
                                recursive=recursive,
                                keep_structure=keep_structure,
                                white_threshold=white_threshold,
                                sep_ratio=sep_ratio,
                                min_seg=min_seg,
                                trim_ratio=trim_ratio,
                                disable_trim=disable_trim,
                                quality_jpg=quality_jpg,
                            )
                        ok += 1
                        self._msg_q.put(("log", f"[OK] {img_path.name} -> {out_folder}（{tiles_count} 张）"))
                    except Exception as e:
                        fail += 1
                        self._msg_q.put(("log", f"[FAIL] {img_path.name}：{e}"))

                    self._msg_q.put(("progress", idx, len(paths)))

                self._msg_q.put(("done", ok, fail))
            except Exception:
                self._msg_q.put(("fatal", traceback.format_exc()))

        def _drain_messages(self) -> None:
            try:
                while True:
                    msg = self._msg_q.get_nowait()
                    kind = msg[0]
                    if kind == "log":
                        self._append_log(msg[1])
                    elif kind == "progress":
                        i, total = msg[1], msg[2]
                        self.progress["value"] = i
                        self.progress_label.configure(text=f"{i}/{total}")
                    elif kind == "done":
                        ok, fail = msg[1], msg[2]
                        self._append_log(f"完成：成功 {ok}，失败 {fail}")
                        self._set_running(False)
                        messagebox.showinfo("完成", f"处理完成：成功 {ok}，失败 {fail}")
                    elif kind == "fatal":
                        self._append_log("发生致命错误：\n" + msg[1])
                        self._set_running(False)
                        messagebox.showerror("错误", "发生致命错误，详见日志。")
            except queue.Empty:
                pass
            finally:
                self.after(120, self._drain_messages)


def main() -> None:
    if tk is None:
        raise SystemExit(
            "当前 Python 环境缺少 tkinter，无法启动桌面图形界面。\n"
            "解决办法：\n"
            "1) Windows/macOS：安装官方 Python（通常自带 tkinter）\n"
            "2) Ubuntu/Debian：sudo apt-get install python3-tk\n"
            "3) 或者我也可以给你做一个浏览器 Web 界面版本（更通用）。"
        )
    app = App()  # type: ignore[name-defined]
    app.mainloop()


if __name__ == "__main__":
    main()
