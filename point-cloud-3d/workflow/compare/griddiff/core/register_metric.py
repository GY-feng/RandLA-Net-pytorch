#-*- encoding:utf-8 -*-
METRIC_REGISTRY = {}

def register_metric(name):
    """装饰器：只打上注册标记"""
    def decorator(func):
        func._metric_name = name  # 给这个函数打标
        return func
    return decorator

