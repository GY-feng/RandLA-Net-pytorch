import yaml
import os
from yacs.config import CfgNode as CN
import sys

module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from workflow.utils.logger import logger

class ConfigNode(CN):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)

    def validate(self):
        """
        校验配置项，确保参数间的约束关系得以满足
        """

        if self.use_cache and not self.cache_dir:
            raise ValueError("缓存路径未指定！")
        
        if self.use_seg_slope and not self.coord_json:
            raise ValueError("管辖区坐标文件未指定！")
        
        if self.icp.to_register and not self.icp.icp_method:
            raise ValueError("具体配准算法未指定！")
        
        if self.save and not self.result_dir:
            raise ValueError("结果保存路径未指定！")

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        r = ''
        s = []
        for k, v in self.items():
            separator = '\n' if isinstance(v, ConfigNode) else ' '
            if isinstance(v, str) and not v:
                v = '\'\''
            attr_str = f'{str(k)}:{separator}{str(v)}'
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += '\n'.join(s)
        return r

    def as_dict(self):
        def convert_to_dict(node):
            if not isinstance(node, ConfigNode):
                return node
            else:
                dic = dict()
                for k, v in node.items():
                    dic[k] = convert_to_dict(v)
                return dic

        return convert_to_dict(self)

    @staticmethod
    def load_from_yaml(yaml_file):
        """
        从 YAML 文件加载配置并返回一个 ConfigNode 实例
        :param yaml_file: YAML 配置文件路径
        :return: ConfigNode 实例
        """
        with open(yaml_file, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        # 将加载的数据传递给 ConfigNode 构造函数
        config_node = ConfigNode(config_data)
        return config_node