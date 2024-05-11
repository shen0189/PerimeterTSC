from peritsc.signal_controller import PeriSignalController
from peritsc.perimeterdata import PeriSignals
from utils.utilize import config


class TimeSlotPeriSignalController(PeriSignalController):

    def __init__(self, peri_data: PeriSignals, action: float):
        super(TimeSlotPeriSignalController, self).__init__(peri_data, action)

    # 重写分布式各交叉口slot-based配时确定方法
    def set_local_green(self):
        '''
        输入：无（交叉口状态）
        固定变量: inflow方向绿灯时长
        决策变量：所有方向绿灯时长+绿灯启亮（+相序）
        目标：最大化吞吐量
        '''
        cycle, yellow = config['cycle_time'], config['yellow_duration']
        M = 1e5
        perimeter_total_throughput = 0  # 返回值


        pass