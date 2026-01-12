import os
import logging
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

@dataclass
class InferenceLog:
    seq_idx: int
    num_images: int
    recv_start_time: float
    recv_end_time: float
    model_start_time: float
    model_end_time: float
    send_start_time: float
    send_end_time: float
    last_frame: np.ndarray
    actions: np.ndarray
    instruction: str
    outputs: str

class InferenceLogger:
    log_format = '%(asctime)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    log_level = logging.INFO
    max_logs = 20

    def __init__(self, save_dir: str):

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        existing_logs = [f for f in os.listdir(self.save_dir) if f.endswith('.txt')]
        if len(existing_logs) >= self.max_logs:
            print(f"[InferenceLogger] Log files exceed {self.max_logs}, clearing old logs...")
            for f in existing_logs:
                os.remove(os.path.join(self.save_dir, f))
                
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        self.log_path = os.path.join(self.save_dir, log_filename)

        self.logger = logging.getLogger(f"InferenceLogger_{log_filename}")
        self.logger.setLevel(self.log_level)

        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
            fh.setLevel(self.log_level)
            # formatter = logging.Formatter(self.log_format, datefmt=self.date_format)
            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        print(f"[InferenceLogger] Logging to: {self.log_path}")

    def log(self, log_entry: InferenceLog):
        recv_time_str = datetime.fromtimestamp(
            log_entry.recv_start_time,
            tz=timezone(timedelta(hours=8))
        ).strftime('%Y-%m-%d %H:%M:%S')

        send_time_str = datetime.fromtimestamp(
            log_entry.send_end_time,
            tz=timezone(timedelta(hours=8))
        ).strftime('%Y-%m-%d %H:%M:%S')

        duration_total = log_entry.send_end_time - log_entry.recv_start_time
        duration_recv = log_entry.recv_end_time - log_entry.recv_start_time
        duration_model = log_entry.model_end_time - log_entry.model_start_time
        duration_send = log_entry.send_end_time - log_entry.send_start_time

        msg = (
            f"[Seq]: {log_entry.seq_idx} | "
            f"[RecvTime]: {recv_time_str} | "
            f"[SendTime]: {send_time_str} | "
            f"[TotalDuration]: {duration_total:.3f}s | "
            f"[RecvDuration]: {duration_recv:.3f}s | "
            f"[ModelDuration]: {duration_model:.3f}s | "
            f"[SendDuration]: {duration_send:.3f}s | "
            f"[Instruction]: {log_entry.instruction} | "
            f"[NumImages]: {log_entry.num_images} | "
            f"[Outputs]: {log_entry.outputs}"
        )
        self.logger.info(msg)

    def get_log_path(self):
        return self.log_path
