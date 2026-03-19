"""
结果可视化类，暂时通过第三方软件进行展示
"""
from collections import defaultdict
import numpy
from .figure import Surface3d


class Visualization:
    def __init__(self, monitor, visualize):
        self.clean_step = 500
        self.monitor = monitor
        self.vis = visualize

    def show(self, step, ext=None):
        """
        1. 获取module_name
        2. 获取module_name 对应的 quantity
        2. 获取结果值 module:quantity:epoch
        :return:
        """
        logs = defaultdict(dict)
        module_names = self._get_module_name()
        for module_name in module_names:
            quantitis = self.monitor.parse_quantity[module_name]
            quantity_names = self._get_quantity_name(module_name)
            for quantity, quantity_name in zip(quantitis, quantity_names):
                if not quantity.should_show(step):
                    continue
                key = module_name + '_' + quantity_name
                val = self._get_result(module_name, quantity_name, step)
                logs.update(self._format_log_value(key, module_name, quantity_name, val))
        if ext is not None:
            logs.update(ext)
        self.vis.log(logs)
        # if step % self.clean_step == 0:
        #     self.monitor.clean_mem()

    
    def log_ext(self, ext=None):
        self.vis.log(ext)

    def close(self):
        self.vis.finish()
        return

    def _get_module_name(self):
        module_names = self.monitor.get_output().keys()
        return module_names

    def _get_quantity_name(self, module_name):
        quantity_name = self.monitor.get_output()[module_name].keys()
        return quantity_name

    def _get_result(self, module_name, quantity_name, step=None):
        if step != None:
            value = self.monitor.get_output()[module_name][quantity_name][step]
        else:
            value = self.monitor.get_output()[module_name][quantity_name]
        return value

    def _format_log_value(self, key, module_name, quantity_name, value):
        if isinstance(value, dict):
            logs = {}
            for child_key, child_value in value.items():
                logs.update(self._format_log_value(f"{key}_{child_key}", module_name, quantity_name, child_value))
            return logs

        if isinstance(value, numpy.ndarray) and value.size == 1:
            return {key: value.item()}

        if isinstance(value, (float, int, numpy.float32, numpy.float64)):
            return {key: value}

        history = self._get_result(module_name, quantity_name)
        return {key: Surface3d(history, key)}
