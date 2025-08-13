import argparse
import threading
import queue
import time
import os
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont

from train.train import load_config
from train.utils import build_model_from_cfg, load_state_dict_forgiving
from data_loader.utils import get_num_classes

# ---------------------------
# Audio + feature processing
# ---------------------------
def record_audio(duration_s: float, sr: int) -> np.ndarray:
    rec = sd.rec(int(duration_s * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return rec[:, 0].copy()

def preprocess_wave(y: np.ndarray, sr: int, cfg_mfcc: dict, fixed_duration_s: float) -> torch.Tensor:
    # Pad / trim to fixed duration
    target_len = int(fixed_duration_s * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    n_mfcc = int(cfg_mfcc["n_mfcc"])
    frame_length_s = float(cfg_mfcc["frame_length_s"])
    hop_length_s   = float(cfg_mfcc["hop_length_s"])
    n_fft = int(round(frame_length_s * sr))
    hop_length = int(round(hop_length_s * sr))

    # Librosa MFCC (match training assumptions)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True
    )  # shape (n_mfcc, T)

    # (C, T) float32 tensor
    feat = torch.from_numpy(mfcc).float()
    return feat  # (channels, time)

def prepare_input_tensor(feat: torch.Tensor) -> torch.Tensor:
    # Add batch dim
    return feat.unsqueeze(0)  # (1, C, T)

# ---------------------------
# Config helpers
# ---------------------------
def get_mfcc_section(cfg: dict) -> dict:
    """
    Locate MFCC settings in config. Supports:
      cfg['mfcc']
      cfg['data']['mfcc']
      cfg['features']['mfcc']
      cfg['audio']['mfcc']
    """
    return (
        cfg.get("mfcc")
        or cfg.get("data", {}).get("mfcc")
        or cfg.get("features", {}).get("mfcc")
        or cfg.get("audio", {}).get("mfcc")
    )

# ---------------------------
# Model wrapper
# ---------------------------
class KeywordModel:
    def __init__(self, cfg_path, weights_path, device):
        self.cfg = load_config(cfg_path)
        # Find MFCC section (now under data.mfcc in base.yaml)
        self.mfcc_cfg = get_mfcc_section(self.cfg)
        if self.mfcc_cfg is None:
            raise KeyError(
                "Could not find 'mfcc' section in config. "
                f"Top-level keys: {list(self.cfg.keys())}"
            )

        self.device = device
        self.task_type = self.cfg["task"]["type"]
        self.class_list = list(self.cfg["task"]["class_list"])
        if self.cfg["task"].get("include_unknown"):
            self.class_list.append("unknown")
        if self.cfg["task"].get("include_background"):
            self.class_list.append("background")
        num_classes = len(self.class_list) if self.task_type == "multiclass" else 1

        sr = int(self.mfcc_cfg["sample_rate"])
        fixed_dur = float(self.mfcc_cfg["fixed_duration_s"])
        dummy = np.zeros(int(sr * fixed_dur), dtype=np.float32)
        feat = preprocess_wave(dummy, sr, self.mfcc_cfg, fixed_dur)
        self.model = build_model_from_cfg(self.cfg, feat, num_classes)
        self.model = load_state_dict_forgiving(self.model, weights_path, device)
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, feat: torch.Tensor, top_k=3):
        x = prepare_input_tensor(feat).to(self.device)
        logits = self.model(x)
        if self.task_type == "multiclass":
            probs = F.softmax(logits, dim=1)[0]
            topk = min(top_k, probs.numel())
            vals, idx = torch.topk(probs, topk)
            results = [(self.class_list[i], float(vals[j])) for j, i in enumerate(idx)]
            return results
        else:
            prob = torch.sigmoid(logits)[0, 0].item()
            return [("keyword", prob), ("non_keyword", 1 - prob)]

    @torch.no_grad()
    def predict_full(self, feat: torch.Tensor):
        x = prepare_input_tensor(feat).to(self.device)
        logits = self.model(x)
        if self.task_type == "multiclass":
            probs = F.softmax(logits, dim=1)[0]
            # Return in original class order (for consistent box layout)
            return [(cls, float(probs[i])) for i, cls in enumerate(self.class_list)]
        else:
            prob = torch.sigmoid(logits)[0, 0].item()
            return [("keyword", prob), ("non_keyword", 1 - prob)]

# ---------------------------
# GUI
# ---------------------------
class App:
    def __init__(self, root, model: KeywordModel):
        self.root = root
        self.model = model
        self.cfg = model.cfg
        self.mfcc_cfg = get_mfcc_section(self.cfg)
        if self.mfcc_cfg is None:
            raise KeyError("MFCC config missing (checked: mfcc / data.mfcc / features.mfcc / audio.mfcc).")
        self.sr = int(self.mfcc_cfg["sample_rate"])
        self.fixed_duration_s = float(self.mfcc_cfg["fixed_duration_s"])

        self.root.title("TCN-KWS")
        self.root.configure(bg="#f3f6fa")
        # Make window taller for full bar chart visibility
        self.root.geometry("780x640")

        # Styles
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Result.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Status.TLabel", foreground="#555")
        style.configure("TButton", padding=6)
        style.configure("BG.TFrame", background="#f3f6fa")
        style.configure("BG.TLabelframe", background="#f3f6fa")
        style.configure("BG.TLabelframe.Label", background="#f3f6fa")

        self.status_var = tk.StringVar(value="Idle")
        self.result_var = tk.StringVar(value="Prediction: –")
        self.duration_var = tk.DoubleVar(value=self.fixed_duration_s)
        self.topk_var = tk.IntVar(value=3)

        outer = ttk.Frame(root, padding=14, style="BG.TFrame")
        outer.pack(fill="both", expand=True)

        # Header
        ttk.Label(outer, text="TCN-KWS", style="Header.TLabel").grid(row=0, column=0, columnspan=6, pady=(0, 8), sticky="w")

        # Controls row
        ttk.Label(outer, text="Record Duration (s):").grid(row=1, column=0, sticky="e", padx=(0,4))
        ttk.Entry(outer, textvariable=self.duration_var, width=7, justify="center").grid(row=1, column=1, sticky="w")
        ttk.Label(outer, text="Top-K:").grid(row=1, column=2, sticky="e", padx=(16,4))
        ttk.Entry(outer, textvariable=self.topk_var, width=5, justify="center").grid(row=1, column=3, sticky="w")

        # Record indicator (blue light when active)
        self.record_canvas = tk.Canvas(outer, width=20, height=20, highlightthickness=0, bd=0)
        self.indicator = self.record_canvas.create_oval(4,4,16,16, fill="#888888", outline="#666666")
        ttk.Label(outer, text="REC").grid(row=1, column=4, sticky="w", padx=(6,0))
        self.record_canvas.grid(row=1, column=5, sticky="w")

        # Buttons row
        btn_frame = ttk.Frame(outer)
        btn_frame.grid(row=2, column=0, columnspan=6, pady=(10, 6), sticky="we")
        for c in range(6):
            outer.columnconfigure(c, weight=1)
        for c in range(4):
            btn_frame.columnconfigure(c, weight=1)

        self.record_btn = ttk.Button(btn_frame, text="● Record", command=self.on_record)
        self.open_btn   = ttk.Button(btn_frame, text="Open WAV", command=self.on_open)
        self.predict_btn= ttk.Button(btn_frame, text="Predict", command=self.on_predict_loaded)
        self.clear_btn  = ttk.Button(btn_frame, text="Clear", command=self.on_clear)

        self.record_btn.grid(row=0, column=0, padx=4, sticky="we")
        self.open_btn.grid(  row=0, column=1, padx=4, sticky="we")
        self.predict_btn.grid(row=0, column=2, padx=4, sticky="we")
        self.clear_btn.grid(  row=0, column=3, padx=4, sticky="we")

        # ---- Class probability boxes (grid) ----
        import math
        self.box_font = tkfont.Font(family="Segoe UI", size=10)
        self.box_font_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        classes_frame = ttk.LabelFrame(outer, text="Classes")
        classes_frame.grid(row=3, column=0, columnspan=6, sticky="nsew", pady=(2,6))
        outer.rowconfigure(3, weight=0)

        self.class_boxes = {}
        n_classes = len(self.model.class_list)
        # Choose number of columns for a balanced grid
        cols = 4 if n_classes > 8 else min(4, n_classes)
        rows = math.ceil(n_classes / cols)
        for r in range(rows):
            classes_frame.rowconfigure(r, weight=1)
        for c in range(cols):
            classes_frame.columnconfigure(c, weight=1)

        for idx, cls_name in enumerate(self.model.class_list):
            r = idx // cols
            c = idx % cols
            lbl = tk.Label(
                classes_frame,
                text=cls_name,
                bd=1,
                relief="ridge",
                width=14,
                padx=4,
                pady=4,
                font=self.box_font,
                bg="#d9d9d9"
            )
            lbl.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
            self.class_boxes[cls_name] = lbl

        # Shift result + top-k + status rows down
        result_row = 4
        topk_row = 5
        status_row = 6

        ttk.Label(outer, textvariable=self.result_var, style="Result.TLabel").grid(
            row=result_row, column=0, columnspan=6, sticky="w", pady=(2,4)
        )

        box_frame = ttk.LabelFrame(outer, text="Class Probabilities", style="BG.TLabelframe")
        box_frame.grid(row=topk_row, column=0, columnspan=6, sticky="nsew", pady=(0,6))
        outer.rowconfigure(topk_row, weight=1)
        self.topk_canvas = tk.Canvas(box_frame, height=260, highlightthickness=0, bd=0, bg="#ffffff")
        self.topk_canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self._topk_canvas_items = []
        # Initialize with zero-probability bars so chart is visible immediately
        self.last_full = [(cls, 0.0) for cls in self.model.class_list]
        self.topk_canvas.bind("<Configure>", self.on_canvas_resize)
        # Draw once geometry is ready
        self.root.after(120, self.redraw_bars)

        status_bar = ttk.Frame(outer)
        status_bar.grid(row=status_row, column=0, columnspan=6, sticky="we")
        ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w")

        # Remove old placement of result/top-k/status (they were previously rows 3,4,5)
        self.audio_buffer = None
        self.pred_queue = queue.Queue()
        self.root.after(100, self.poll_queue)

    def set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def set_record_indicator(self, recording: bool):
        self.record_canvas.itemconfig(
            self.indicator,
            fill="#1E90FF" if recording else "#888888",
            outline="#0F4C92" if recording else "#666666"
        )

    def on_record(self):
        dur = float(self.duration_var.get())
        self.set_status(f"Recording {dur:.2f}s ...")
        self.set_record_indicator(True)
        self.record_btn.config(state="disabled")
        self.predict_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        t = threading.Thread(target=self._record_thread, args=(dur,))
        t.daemon = True
        t.start()

    def _record_thread(self, dur):
        try:
            audio = record_audio(dur, self.sr)
            self.audio_buffer = audio
            self.pred_queue.put(("status", f"Recorded {len(audio)/self.sr:.2f}s"))
            self.pred_queue.put(("predict", None))
        except Exception as e:
            self.pred_queue.put(("error", str(e)))
        finally:
            self.pred_queue.put(("record_done", None))

    def on_open(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        try:
            y, sr = librosa.load(path, sr=self.sr, mono=True)
            self.audio_buffer = y
            self.set_status(f"Loaded: {os.path.basename(path)} ({len(y)/sr:.2f}s)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    def on_clear(self):
        self.audio_buffer = None
        self.result_var.set("Prediction: –")
        self.clear_topk_canvas()
        self.set_status("Cleared buffer.")

    def on_predict_loaded(self):
        if self.audio_buffer is None:
            messagebox.showwarning("No audio", "Record or load audio first.")
            return
        self.predict_current()

    def reset_class_boxes(self):
        for cls, lbl in self.class_boxes.items():
            lbl.config(bg="#d9d9d9", font=self.box_font)

    def shade_class_boxes(self, full_probs):
        # full_probs: list of (cls, prob)
        # Map prob to blended color between grey and green
        def prob_to_color(p: float):
            base = (217, 217, 217)      # #d9d9d9
            hi   = (46, 180, 80)        # greenish
            r = int(base[0] + (hi[0]-base[0]) * p)
            g = int(base[1] + (hi[1]-base[1]) * p)
            b = int(base[2] + (hi[2]-base[2]) * p)
            return f"#{r:02x}{g:02x}{b:02x}"

        for cls, p in full_probs:
            if cls in self.class_boxes:
                self.class_boxes[cls].config(bg=prob_to_color(p), font=self.box_font)

    def highlight_winner(self, winner_cls: str):
        if winner_cls in self.class_boxes:
            self.class_boxes[winner_cls].config(font=self.box_font_bold)

    def predict_current(self):
        try:
            feat = preprocess_wave(
                self.audio_buffer, self.sr,
                self.mfcc_cfg, self.fixed_duration_s
            )
            full = self.model.predict_full(feat)
            self.update_results(full)  # also triggers bar draw
            self.shade_class_boxes(full)
            winner = max(full, key=lambda x: x[1])[0]
            self.highlight_winner(winner)
            # Keep highlighted for 5 seconds instead of 1
            self.root.after(5000, self.reset_class_boxes)
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {e}")

    def clear_topk_canvas(self):
        self.topk_canvas.delete("all")
        self._topk_canvas_items = []

    def on_canvas_resize(self, event):
        # Redraw bars on any size change
        if self.last_full:
            self.draw_prob_bars(self.last_full)

    def draw_prob_bars(self, results):
        """
        Vertical bar chart: bar height = probability, color intensity = probability.
        results: list of (label, prob) in fixed original order.
        """
        if not results:
            self.clear_topk_canvas()
            return
        # Make sure geometry is updated
        self.topk_canvas.update_idletasks()
        w = self.topk_canvas.winfo_width()
        h = self.topk_canvas.winfo_height()
        if w < 80 or h < 80:
            # Retry shortly until canvas has a real size
            self.topk_canvas.after(60, lambda r=results: self.draw_prob_bars(r))
            return
        self.clear_topk_canvas()
        pad_x = 18
        pad_y = 18
        n = len(results)
        gap = 12
        total_gap = gap * (n - 1)
        avail_w = max(10, w - 2 * pad_x - total_gap)
        bar_w = avail_w / n
        max_bar_h = max(10, h - 2 * pad_y - 22)  # leave space for labels
        base = (220, 220, 220)     # grey
        hi = (46, 180, 80)         # green
        for i, (lbl, p) in enumerate(results):
            x0 = pad_x + i * (bar_w + gap)
            x1 = x0 + bar_w
            bar_h = p * max_bar_h
            if bar_h < 1 and p == 0:
                bar_h = 1  # minimal visible line for zero prob
            y1 = h - pad_y
            y0 = y1 - bar_h
            # Color blend
            r = int(base[0] + (hi[0]-base[0]) * p)
            g = int(base[1] + (hi[1]-base[1]) * p)
            b = int(base[2] + (hi[2]-base[2]) * p)
            color = f"#{r:02x}{g:02x}{b:02x}"
            # Background track (light grey)
            self.topk_canvas.create_rectangle(x0, y1 - max_bar_h, x1, y1, fill="#f0f0f0", width=0)
            # Bar
            self.topk_canvas.create_rectangle(x0, y0, x1, y1, fill=color, width=0)
            # Probability text (inside or above)
            prob_txt = f"{p:.1%}"
            txt_y = y0 - 12
            if txt_y < 4:
                txt_y = y0 + 12
            self.topk_canvas.create_text((x0 + x1)/2, txt_y, text=prob_txt,
                                         font=("Segoe UI", 9), fill="#333")
            # Label below
            self.topk_canvas.create_text((x0 + x1)/2, y1 + 10, text=lbl,
                                         font=("Segoe UI", 9), anchor="n")
        self.topk_canvas.config(scrollregion=self.topk_canvas.bbox("all"))

    def redraw_bars(self):
        if self.last_full:
            self.draw_prob_bars(self.last_full)

    def update_results(self, full_results):
        if not full_results:
            self.result_var.set("Prediction: -")
            return
        # Determine winner (max prob)
        winner_label, winner_prob = max(full_results, key=lambda x: x[1])
        self.result_var.set(f"Prediction: {winner_label} ({winner_prob:.2%})")
        self.last_full = full_results
        # Draw / refresh bar chart immediately
        self.draw_prob_bars(full_results)

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.pred_queue.get_nowait()
                if kind == "status":
                    self.set_status(payload)
                elif kind == "predict":
                    self.set_status("Predicting...")
                    self.predict_current()
                    self.set_status("Done")
                elif kind == "record_done":
                    self.set_record_indicator(False)
                    self.record_btn.config(state="normal")
                    self.predict_btn.config(state="normal")
                    self.open_btn.config(state="normal")
                elif kind == "error":
                    messagebox.showerror("Error", payload)
                    self.set_record_indicator(False)
                    self.record_btn.config(state="normal")
                    self.predict_btn.config(state="normal")
                    self.open_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

# ---------------------------
# Main
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", choices=["cpu","cuda","auto"], default="auto")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device in ("cuda","auto") and torch.cuda.is_available()) else "cpu")
    model_wrap = KeywordModel(args.config, args.weights, device)
    root = tk.Tk()
    App(root, model_wrap)
    root.mainloop()

if __name__ == "__main__":
    main()