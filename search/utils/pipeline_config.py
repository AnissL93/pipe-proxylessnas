
class PipelineConfig(object):

    BIG_CORE = "big"
    SMALL_CORE = "small"
    GPU = "gpu"

    def __init__(self) -> None:
        # [0,10) -> big
        # [10,20) -> small
        self.mapping = {}
        pass

    def total_stage(self):
        return 25

    def add_stage_mapping(self, key, core):
        self.mapping[key] = core

    @property
    def split_points(self):
        return list(self.mapping.keys())

    def get_proc(self, sp):
        return self.mapping[sp]

    def get_stage(self, b):
        for sp in self.split_points:
            if b >= sp[0] and b < sp[1]:
                return sp

        return None

    def get_proc_of_block(self, idx):
        for (st, ed), val in self.mapping.items():
            if idx >= st and idx < ed:
                return val

        return None

    
def cpu_config(sps):
    a = PipelineConfig()
    a.add_stage_mapping((0, sps[0]), PipelineConfig.BIG_CORE)
    a.add_stage_mapping((sps[0], 25), PipelineConfig.SMALL_CORE)
    return a

def cpu_gpu_config(sps):
    sp1 = sps[0]
    sp2 = sps[1]
    a = PipelineConfig()
    a.add_stage_mapping((0, sp1), PipelineConfig.BIG_CORE)
    a.add_stage_mapping((sp1, sp2), PipelineConfig.SMALL_CORE)
    a.add_stage_mapping((sp2, 25), PipelineConfig.GPU)
    return a


if __name__ == "_main_":
    a = PipelineConfig()
    a.add_stage_mapping((0, 15), "big")
    a.add_stage_mapping((15, 25), "small")

    print(a.get_proc_of_block(3))
    print(a.get_proc_of_block(15))


