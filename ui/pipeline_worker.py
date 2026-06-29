"""流水线工作线程 + 通用后台任务线程 + 共享错误处理"""

from __future__ import annotations

from typing import TYPE_CHECKING
import os
from PySide6.QtCore import Signal, QThread

if TYPE_CHECKING:
    from core.pipeline import PipelineContext


class TaskThread(QThread):
    """通用后台任务线程：运行带 done_cb 回调的阻塞函数

    用法:
        t = TaskThread(target=some_blocking_func,
                       args=[arg1, arg2],
                       kwargs={'key': value},
                       log_cb=self.log)
        t.done.connect(on_result)
        t.start()
        # done 信号携带 (status: str, *args)
    """
    done = Signal(object)  # (status_str, *result_args)

    def __init__(self, target, args=None, kwargs=None, log_cb=None, parent=None):
        super().__init__(parent)
        self._target = target
        self._args = args or []
        self._kwargs = kwargs or {}
        self._log_cb = log_cb
        self._result = ()  # 空元组,防止 done_cb 未被调用时 *self._result 报错

    def run(self):
        def _done_cb(*result_args):
            self._result = result_args

        try:
            self._target(
                *self._args,
                **self._kwargs,
                log_cb=self._log_cb,
                done_cb=_done_cb,
            )
            self.done.emit(('ok', *self._result))
        except Exception as e:
            self.done.emit(('error', str(e)))


class _WorkerErrorMixin:
    """Worker 线程共享的错误处理和配置刷新同步机制"""

    def _init_sync_primitives(self):
        from PySide6.QtCore import QMutex, QWaitCondition
        self._error_mutex = QMutex()
        self._error_cond = QWaitCondition()
        self._error_response = None
        self._config_mutex = QMutex()
        self._config_cond = QWaitCondition()
        self._config_refreshed = False
        self._cancelled = False

    def _wait_for_user_choice(self):
        """阻塞等待主线程对话框返回用户选择"""
        self._error_mutex.lock()
        self._error_response = None
        while self._error_response is None and not self._cancelled:
            self._error_cond.wait(self._error_mutex)
        if self._cancelled:
            self._error_mutex.unlock()
            return None
        choice = self._error_response
        self._error_response = None
        self._error_mutex.unlock()
        return choice

    def _refresh_config(self, log_cb):
        """重试前请求主线程刷新配置"""
        self._config_mutex.lock()
        self._config_refreshed = False
        self.refresh_ctx_config.emit()
        if not self._config_cond.wait(self._config_mutex, 30000):
            log_cb("  ⚠️ 等待配置刷新超时,使用旧配置重试")
        self._config_mutex.unlock()

    def _handle_tts_errors(self, failures, ctx, syn_fn, cfg_fn):
        """TTS 错误处理：弹窗 → 重试3次/跳过/终止"""
        from core.pipeline import CancelledError
        for idx, error_msg, text in failures:
            self.tts_error.emit(idx, error_msg, text)
            choice = self._wait_for_user_choice()
            if choice is None:
                return
            if choice == "abort":
                ctx.cancelled = True
                raise CancelledError()
            elif choice == "retry":
                sub = next((s for s in ctx.subs if s.idx == idx), None)
                if sub is None:
                    continue
                retry_ok = False
                for attempt in range(3):
                    try:
                        self._refresh_config(ctx.log_ui)
                        tts_path = syn_fn(
                            sub,
                            cache=ctx.cache, work_dir=ctx.work_dir,
                            tts_cfg=cfg_fn(ctx),
                            log_cb=ctx.log_ui, check_cancelled=ctx.check_cancelled,
                        )
                        if tts_path:
                            self.tts_item_ready.emit(idx, tts_path)
                            ctx.log_ui(f"  🎯 第{idx}条 重试成功: {tts_path}")
                            retry_ok = True
                            break
                    except Exception as retry_e:
                        ctx.log_ui(f"  ⚠️ 第{idx}条 第{attempt+1}次重试失败: {retry_e}")
                        if attempt < 2:
                            self.tts_error.emit(idx, str(retry_e), sub.text)
                            choice = self._wait_for_user_choice()
                            if choice is None:
                                return
                            if choice == "abort":
                                ctx.cancelled = True
                                raise CancelledError()
                            elif choice == "skip":
                                break
                if not retry_ok:
                    ctx.log_ui(f"  ⚠️ 第{idx}条 多次重试失败,已跳过")


class PipelineWorker(_WorkerErrorMixin, QThread):
    """流水线/单步执行工作线程 — step_idx=None 执行全流程,否则只执行单步"""

    # ── 公共信号 ──
    log_msg = Signal(str)
    step_progress = Signal(int, int, str)
    finished = Signal(bool, str)
    tts_error = Signal(int, str, str)
    tts_item_ready = Signal(int, str)
    refresh_ctx_config = Signal()

    # ── 全长流水线专用信号 ──
    subs_calib_gender_ready = Signal(list)
    calib_src_ready = Signal(list)
    ref_audio_ready = Signal(int, str)
    mix_audio_ready = Signal(int, str)
    tts_cache_hit = Signal(int)
    mix_item_ready = Signal(list)
    mixed_audio_ready = Signal(str)

    def __init__(self, ctx: 'PipelineContext', step_idx: int = None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._step_idx = step_idx
        self._init_sync_primitives()

    def cancel(self):
        self._cancelled = True
        self.ctx.cancelled = True

    def run(self):
        self._calib_src_emitted = False  # 重置,允许新流程发出信号
        from core.pipeline import PipelineOrchestrator, CancelledError, synthesize_tts_segment, _ctx_to_tts_config

        self.ctx.log_ui = self.log_msg.emit
        is_full = self._step_idx is None

        def tts_cb(idx, path):
            return self.tts_item_ready.emit(idx, path)
        def mix_cb(ids):
            return self.mix_item_ready.emit(ids)
        self.ctx.on_tts_error_cb = lambda failures: self._handle_tts_errors(
            failures, self.ctx, synthesize_tts_segment, _ctx_to_tts_config
        )
        self.ctx.on_subs_ready = lambda subs: self.subs_calib_gender_ready.emit(subs)
        self.ctx.on_raw_subs_ready = lambda raw_subs: self.calib_src_ready.emit(raw_subs)
        self.ctx.on_audio_ready = lambda path: self.mix_audio_ready.emit(0, path)
        self.ctx.on_vocals_ready = lambda path: self.ref_audio_ready.emit(0, path)
        self.ctx.on_mix_done = lambda path: self.mixed_audio_ready.emit(path)
        self.ctx.progress_cb = self._on_progress_step
        orch = PipelineOrchestrator(self.ctx, tts_item_cb=tts_cb, mix_item_cb=mix_cb)

        try:
            if is_full:
                orch.run()
                if self._cancelled:
                    self.finished.emit(False, "已取消")
                else:
                    self.finished.emit(True, "配音完成")
            else:
                orch.run_single_step(self._step_idx)
                self.finished.emit(True, f"Step {self._step_idx+1} 完成")
        except CancelledError:
            self.finished.emit(False, "已取消")
        except RuntimeError as e:
            self.ctx.log_ui(str(e))
            self.finished.emit(False, str(e))
        except Exception as e:
            import traceback
            self.ctx.log_ui(traceback.format_exc())
            self.finished.emit(False, str(e))

    def _on_progress_step(self, step_idx, pct, text):
        if self._cancelled:
            return
        self.step_progress.emit(step_idx, pct, text)
        if pct != 100:
            return

        if step_idx == 2 and self.ctx.raw_subs:
            if getattr(self, '_calib_src_emitted', False):
                return
            self._calib_src_emitted = True
            self.ctx.log_ui(f"  发出 calib_src_ready: {len(self.ctx.raw_subs)} 条")
            self.calib_src_ready.emit(self.ctx.raw_subs)
        elif step_idx == 3:
            if self.ctx.cache and self.ctx.subs:
                for sub in self.ctx.subs:
                    if os.path.exists(self.ctx.cache.tts_path(sub)):
                        self.tts_cache_hit.emit(sub.idx)
