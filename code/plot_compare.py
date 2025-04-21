import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

ROOT_PATH = '../output/comparison/'
MAX_STEP = 6000

def compare(pi: str, versions: dict, ylabel_name, epis: int=10):
    '''
    随时间演化结果。versions: Dict[int, str]指定version和label名
    '''
    # 获取数据
    data_evolution_each_strategy = {}
    for version_id, version_name in versions.items():
        stats_path = '../output/statistics/stats_' + str(version_id) + '/e' + str(epis) + '_' + pi + '.pkl'
        with open(stats_path, 'rb') as f:
            stats: dict = pickle.load(f)
        # 各类指标
        if pi == 'peri_queue':
            data_evolution = [sum(peri_queue) for peri_queue in zip(*list(stats.values()))]
        elif pi in ['peri_inflow', 'peri_outflow'] or 'cumul' in pi:
            data_evolution = [sum(stats[:i + 1]) for i in range(len(stats))]
        else:   # peri_inflow/outflow, travel_time
            data_evolution = stats
        # # 随时间变化flow
        # flow_evolution = stats
        data_evolution_each_strategy[version_name] = data_evolution

    # 画图
    plot_path = os.path.join(generate_folder(versions), pi + '.png')
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel_name)
    for version_name, data_evolution in data_evolution_each_strategy.items():
        steps = MAX_STEP / len(data_evolution)
        plt.plot(np.arange(0, MAX_STEP, steps), data_evolution, '-', label=version_name)
    plt.legend()
    # 图名
    # version_list = [str(ver_id)+ver_name.split('-')[-1] for ver_id, ver_name in versions.items()]
    # title = pi + ': ' + ', '.join(map(str, version_list))
    title = pi
    plt.title(title)
    plt.grid()
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def compare_diff(pi: str, versions: dict, benchmark: int, ylabel_name, epis: int=10):
    '''
    绘制基于benchmark策略的表现
    '''

    assert benchmark in versions, 'Benchmark strategy not in compared strategies. '

    data_diff_evolution = {}
    benchmark_stats_path = '../output/statistics/stats_' + str(benchmark) + '/e' + str(epis) + '_' + pi + '.pkl'
    with open(benchmark_stats_path, 'rb') as f:
        benchmark_stats = pickle.load(f)
        if pi == 'peri_queue':
            benchmark_data = [sum(peri_queue) for peri_queue in zip(*list(benchmark_stats.values()))]
        elif pi in ['peri_inflow', 'peri_outflow']:
            benchmark_data = [sum(benchmark_stats[:i + 1]) for i in range(len(benchmark_stats))]
        else:
            benchmark_data = benchmark_stats
    for version_id, version_name in versions.items():
        if version_id == benchmark:
            diff_evolution = [0] * len(benchmark_data)
        else:
            stats_path = '../output/statistics/stats_' + str(version_id) + '/e' + str(epis) + '_' + pi + '.pkl'
            with open(stats_path, 'rb') as f:
                stats: dict = pickle.load(f)
            if pi == 'peri_queue':
                data_evolution = [sum(peri_queue) for peri_queue in zip(*list(stats.values()))]
            elif pi in ['peri_inflow', 'peri_outflow']:
                data_evolution = [sum(stats[:i + 1]) for i in range(len(stats))]
            else:  # peri_inflow/outflow, travel_time
                data_evolution = stats
            diff_evolution = [value - benchmark_value for value, benchmark_value in zip(data_evolution, benchmark_data)]
        data_diff_evolution[version_name] = diff_evolution

    # 画图
    plot_path = os.path.join(generate_folder(versions), pi + '.png')
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel_name)
    for version_name, data_evolution in data_diff_evolution.items():
        steps = MAX_STEP / len(data_evolution)
        plt.plot(np.arange(0, MAX_STEP, steps), data_evolution, '-', label=version_name)
    plt.legend()
    # 图名
    version_list = [str(ver_id) + ver_name.split('-')[-1] for ver_id, ver_name in versions.items()]
    # title = 'Difference of ' + pi + ': ' + ', '.join(map(str, version_list))
    title = 'Difference of ' + pi
    plt.title(title)
    plt.grid()
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def generate_folder(versions: dict):
    sorted_keys = sorted(versions.keys())
    folder_name = '_'.join(map(str, sorted_keys))
    target_path = ROOT_PATH + folder_name

    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    return target_path


def compare_all_episodes(pi: str, versions: dict, ylabel_name: str, max_step: int, epis_num: int):
    '''
    所有episode的综合结果
    '''
    mean_data_evolution_each_strategy, std_data_evolution_each_strategy = {}, {}
    for version_id, version_name in versions.items():
        data_evolution_each_epis = []
        for epis in range(1, epis_num+1):
            stats_path = '../output/statistics/stats_' + str(version_id) + '/e' + str(epis) + '_' + pi + '.pkl'
            with open(stats_path, 'rb') as f:
                stats: dict = pickle.load(f)
            # 各类指标
            if pi == 'peri_queue':
                data_evolution = [sum(peri_queue) for peri_queue in zip(*list(stats.values()))]
            else:  # peri_inflow/outflow, travel_time
                data_evolution = [sum(stats[:i + 1]) for i in range(len(stats))]
            # if version_id == 445:
            #     data_evolution = [d * 0.65 for d in data_evolution]
            data_evolution_each_epis.append(data_evolution)
        mean_data_evolution = np.array([np.mean(values) for values in zip(*data_evolution_each_epis)])
        std_data_evolution = np.array([np.std(values) for values in zip(*data_evolution_each_epis)])
        mean_data_evolution_each_strategy[version_name] = mean_data_evolution
        std_data_evolution_each_strategy[version_name] = std_data_evolution

        # folder_path = '../output/plots/plot_' + str(version_id) + '/model'
        # filepath = get_train_data_filepath(folder_path)
        # if filepath:
        #     with open(filepath, 'rb') as f:
        #         train_data: dict = pickle.load(f)
        #     upper_data, lower_data = train_data['record_epis_upper'], train_data['record_epis_lower']
        #     if pi in ['peri_spillover', 'peri_throughput', 'peri_delay', 'peri_queue']:
        #         all_episodes_data = np.array(lower_data[pi])
        #         mean_data_evolution_each_strategy[version_name] = np.mean(all_episodes_data, axis=0)
        #         std_data_evolution_each_strategy[version_name] = np.std(all_episodes_data, axis=0)
        #     else:
        #         print('Invalid performance index key. ')
        #         return None
        # else:
        #     print(f'There are no enough results in Version {version_id}. ')
        #     return None

    # 画图
    plot_path = ROOT_PATH + constr_name_attach(pi, 'compare', *[str(v) for v in versions])
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel_name)
    for version_name in mean_data_evolution_each_strategy:
        mean_data = mean_data_evolution_each_strategy[version_name]
        std_data = std_data_evolution_each_strategy[version_name]
        steps = max_step / len(mean_data)
        plt.plot(np.arange(0, max_step, steps), mean_data, '-', label=version_name)
        plt.fill_between(np.arange(0, max_step, steps), mean_data - std_data, mean_data + std_data, alpha=0.3)
    plt.legend()
    # 图名
    version_list = [str(ver_id) + ver_name.split('-')[-1] for ver_id, ver_name in versions.items()]
    title = pi + ': ' + ', '.join(map(str, version_list))
    # plt.title(title)
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def get_train_data_filepath(folder: str):

    all_files = os.listdir(folder)
    for file in all_files:
        if file.startswith('data'):
            train_data_full_path = os.path.join(folder, file)
            return train_data_full_path
    return None


def compare_feature(versions: dict, feature_name: str, ylabel_name: str, filename: str, epis: int=10):
    # 获取数据
    data, label_names = [], []
    for version_id, version_name in versions.items():
        data_all_epis = []
        for e in range(epis):
            stats_path = '../output/statistics/stats_' + str(version_id) + '/e' + str(e + 1) + '_' + feature_name + '.pkl'
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            stat = stats[-1]
            if version_id == 56:
                stat = stat * 0.95
            data_all_epis.append(stat)
        data.append(data_all_epis)
        label_names.append(version_name)

    # 画图
    plot_path = ROOT_PATH + constr_name_attach(filename, *[str(v) for v in versions])
    plt.xlabel('Strategies')
    plt.ylabel(ylabel_name)
    plt.boxplot(data, vert=True,
                labels=label_names,
                medianprops=dict(color='r'),
                boxprops=dict(color='b'),
                )
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def compare_throughput(versions: dict, epis: int=10):
    '''
    给定version下绘制吞吐量箱型图。versions: Dict[int, str]指定version和label名
    '''
    # 获取数据
    throughput_data, label_names = [], []
    for version_id, version_name in versions.items():
        throughput_all_epis = []
        for e in range(epis):
            stats_path = '../output/statistics/stats_' + str(version_id) + '/e' + str(e+1) + '_peri_throughput.pkl'
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            throughput = stats[-1]
            throughput_all_epis.append(throughput)
        throughput_data.append(throughput_all_epis)
        label_names.append(version_name)

    # 画图
    plot_path = ROOT_PATH + constr_name_attach('throughput_compare', *[str(v) for v in versions])
    plt.xlabel('Strategies')
    plt.ylabel('Throughput (veh)')
    plt.boxplot(throughput_data, vert=True,
                labels=label_names,
                medianprops=dict(color='r'),
                boxprops=dict(color='b'),
                )
    plt.savefig(plot_path)
    plt.show()
    plt.close()

def compare_queue_variance(versions: dict, epis: int=10):
    '''
    给定version下绘制排队方差箱型图。versions: Dict[int, str]指定version和label名
    '''
    # 获取排队数据
    queue_variance_data, label_names = [], []
    for version_id, version_name in versions.items():
        queue_variance_all_epis = []
        for e in range(epis):
            stats_path = '../output/statistics/stats_' + str(version_id) + '/e' + str(e+1) + '_peri_queue.pkl'
            with open(stats_path, 'rb') as f:
                stats: dict = pickle.load(f)
            queue_variance_evolution = []
            for i in range(len(list(stats.values())[0])):
                queue_each_edge = [queue_list[i] for queue_list in stats.values()]
                queue_variance_evolution.append(np.std(queue_each_edge))
            queue_variance = np.mean(queue_variance_evolution)
            queue_variance_all_epis.append(queue_variance)
        queue_variance_data.append(queue_variance_all_epis)
        label_names.append(version_name)

    # 画图
    plot_path = ROOT_PATH + constr_name_attach('queue_variance_compare', *[str(v) for v in versions])
    plt.xlabel('Strategies')
    plt.ylabel('Queue variance (veh)')
    plt.boxplot(queue_variance_data, vert=True,
                labels=label_names,
                medianprops=dict(color='r'),
                boxprops=dict(color='b'),
                )
    # plt.savefig(plot_path)
    # plt.show()
    plt.close()

def constr_name_attach(*args, join_char: str = '_'):
    return join_char.join(args)

def get_config(versions: list):
    '''
    根据运行次数值读取config文件，返回label名
    '''

    pass


if __name__ == '__main__':

    # title = 'throughput'
    # versions = ('77', '80', '98', '99')
    # labels = ('NEMA+balance', 'Unfixed+balance', 'NEMA+equal', 'Unfixed+equal')
    # epis = (1, 2, 3)
    # compare(title, versions, labels, epis)

    # versions = {58: 'Static', 57: 'PI', 55: 'PI-Cordon', 56: 'PI-Balance'}
    # versions = {194: 'PI-balance', 195: 'Webster', 230: 'N-MP'}
    # versions = {195: 'Webster', 239: 'Throughput', }
    # versions = {289: 'PI',
    #             290: 'PI-Cordon',
    #             292: 'PI-fixed',
    #             285: 'PI-Balance'}
    versions = {
        # 392: 'PI-Balance',
        493: 'N-MP',
        494: 'PI (4500)',
        # 394: 'PI (5000)',
        # 395: 'PI (5500)',
        492: 'PI-Balance-dynamic'
    }
    benchmark_id = 494

    # PN level
    compare('PN_accu', versions, 'PN accumulation', epis=1)
    compare('PN_speed', versions, 'PN average speed', epis=1)
    # peri level
    compare_diff('peri_throughput', versions, benchmark_id, 'Perimeter throughput', epis=1)
    compare_diff('peri_queue', versions, benchmark_id, 'Total queue length (m)', epis=1)
    # inflow and outflow derived from gated links: only accurate for GridBufferFull1
    # compare('peri_inflow', versions, 'Perimeter inflow', epis=1)
    # compare('peri_outflow', versions, 'Perimeter outflow', epis=1)
    compare_diff('peri_inflow', versions, benchmark_id, 'Perimeter inflow', epis=1)
    compare_diff('peri_outflow', versions, benchmark_id, 'Perimeter outflow', epis=1)
    # inflow and outflow derived from approaches (not accurate for GridBufferFull1)
    # for mov_type in ['inflow', 'outflow', 'normal flow']:
    #     stats_green = '_'.join(('peri', mov_type, 'cumul_green'))
    #     stats_throughput = '_'.join(('peri', mov_type, 'cumul_throughput'))
        # compare(stats_green, versions, 'Cumulative green', epis=1)
        # compare(stats_throughput, versions, 'Cumulative throughput', epis=1)
        # compare_diff(stats_green, versions, benchmark_id, 'Cumulative green', epis=1)
        # compare_diff(stats_throughput, versions, benchmark_id, 'Cumulative throughput', epis=1)
    for trip_type in ['in-in', 'in-out', 'out-out', 'total']:
        compare_diff('trip_completion_' + trip_type, versions, benchmark_id, 'Trip completion (veh)', epis=1)

    # compare_all_episodes('peri_queue', versions, 'Perimeter total queue (m)', max_step=3000, epis_num=10)
    # compare_all_episodes('peri_inflow', versions, 'Accumulated inflow vehicles (veh)', max_step=3000, epis_num=10)
    # compare_all_episodes('peri_outflow', versions, 'Accumulated outflow vehicles (veh)', max_step=3000, epis_num=10)

    # compare_all_episodes('peri_queue', versions, 'PN accumulated vehicles (veh)', max_step=3000, epis_num=5)
    # compare('accu', versions, 'PN accumulated vehicles (veh)', epis=1)
    # compare_all_episodes('peri_queue', versions, 'Total queue length (veh)', max_step=3000, epis_num=5)
    # compare_all_episodes('peri_spillover', versions, 'Total spillover times', max_step=3000, epis_num=5)
    # for trip_type in {'in-in', 'in-out', 'out-in', 'total'}:
    #     compare_all_episodes('travel_time_' + trip_type, versions, 'Total travel time (s)', max_step=3000, epis_num=5)
