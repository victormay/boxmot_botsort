# boxmot_botsort
## 非常重要
本项目来自**boxmot**,感谢**mikel-brostrom**的杰出工作，以及众多贡献者。  
## 项目说明
项目大部分代码直接来自**boxmot**项目，只对导出部分/reid后端/Strack和BoTSORT类进行了代码调整和更改，由于此部分更改只是方便自己使用，所以重新开了一个自用项目。推荐使用原版https://github.com/mikel-brostrom/boxmot  
## mot
BotSORT
## reid
访问https://github.com/mikel-brostrom/boxmot/blob/master/boxmot/appearance/reid_model_factory.py
## reid后端
onnxruntime-cpu, openvino, torch
## Strack
Strack部分添加了开始时间start_ts以及结束时间end_ts以及存在的时间统计属性track_ts  
类型更新函数update_cls改动  
## BoTSORT
可以像原版正常使用，如果想使用Strack的track_ts属性，需要在update函数中传入ts参数ts为单位为秒类型为浮点数的时间戳  
def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, ts: float = 0) -> np.ndarray:

## 最后
本项目来自**boxmot**,感谢**mikel-brostrom**的杰出工作，以及众多贡献者。  
推荐大家使用原版https://github.com/mikel-brostrom/boxmot  
