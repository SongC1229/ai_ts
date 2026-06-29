"""ML 模型单例生命周期管理基类

为 WavLM 性别检测和 Qwen3 强制对齐两个模型提供统一的生命周期接口：
- load(device, dtype) / unload() / is_loaded()
- 内部由 _load_impl() 子类化实现各自加载逻辑

设计目标：
1. 消除 gender_wavlm.py 与 aligner_qwen.py 之间重复的
   `_model = None / _load_model() / unload_model()` 三段式样板
2. 统一显存释放策略（move_to_cpu + del + gc.collect + cuda.empty_cache)
3. 保留每个模型的兼容入口函数（子类暴露 load_model / unload_model)
"""

from .utils import cleanup_cuda


class MLModelHolder:
    """ML 模型单例生命周期管理基类（类变量共享,无需实例化)

    子类需实现 `_load_impl(cls, device, dtype)` 完成实际模型加载并返回模型对象。
    子类还可重写 `_unload_impl(cls)` 自定义额外的释放逻辑。
    """

    _model = None
    _device = None
    _dtype = None

    @classmethod
    def load(cls, device=None, dtype=None):
        """加载模型（已加载则直接返回单例)

        Args:
            device: 设备字符串（None 表示由子类决定默认值)
            dtype: 精度字符串（None 表示由子类决定默认值)

        Returns:
            模型实例
        """
        if cls._model is not None:
            return cls._model
        cls._device = device
        cls._dtype = dtype
        cls._model = cls._load_impl(cls._device, cls._dtype)
        return cls._model

    @classmethod
    def unload(cls, move_to_cpu=True):
        """卸载模型,释放显存

        Args:
            move_to_cpu: 是否先将模型 .cpu() 再删除
        """
        if cls._model is None:
            return
        try:
            if move_to_cpu and hasattr(cls._model, 'cpu'):
                cls._model = cls._model.cpu()
        except Exception:
            pass
        # 子类自定义释放逻辑
        try:
            cls._unload_impl()
        except NotImplementedError:
            pass
        except Exception:
            pass
        del cls._model
        cls._model = None
        cls._device = None
        cls._dtype = None
        cleanup_cuda()

    @classmethod
    def is_loaded(cls) -> bool:
        """模型是否已加载"""
        return cls._model is not None

    @classmethod
    def _load_impl(cls, device, dtype):
        """子类实现：实际加载并返回模型对象"""
        raise NotImplementedError

    @classmethod
    def _unload_impl(cls):
        """子类可选实现：额外的释放逻辑（默认无操作)"""
        raise NotImplementedError
