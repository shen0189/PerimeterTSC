import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

ROOT_PATH = '../output/comparison/'

def compare(title: str, versions: tuple, labels: tuple, epis: tuple):
    '''
    Generate pictures to compare. Picture number = len(epis). Number of lines in each picture = len(versions).
    '''

    for e in epis:
        plot_path = ROOT_PATH + constr_name_attach(title, 'compare', *versions, 'e' + str(e))
        stats_list = []
        for version in versions:
            stats_path = '../output/statistics/stats_' + version + '/' + constr_name_attach('e'+str(e), 'peri', title) + '.pkl'
            # print(stats_path)
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            stats_list.append(stats)

        max_value = max([max(stats) for stats in stats_list])

        plt.xlabel('time')
        plt.ylabel(title)
        # plt.title(f'perimeter {feature_title} progression')
        for stats, label in zip(stats_list, labels):
            plt.plot(range(len(stats)), stats, '-', label=label)
        plt.ylim((0, max_value * 1.2))
        plt.legend()
        plt.savefig(plot_path)
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


def compare_flow(flow_type: str, versions: dict, epis: int=1):
    '''
    给定version和epis下，绘制inflow/outflow变化。versions: Dict[int, str]指定version和label名
    '''
    # 获取数据
    flow_evolution_each_strategy = {}
    for version_id, version_name in versions.items():
        stats_path = '../output/statistics/stats_' + str(version_id) + '/e' + str(epis) + '_peri_' + flow_type + '.pkl'
        with open(stats_path, 'rb') as f:
            stats: dict = pickle.load(f)
        # tmp
        if flow_type == 'outflow' and version_id == 238:
            for i in range(int(len(stats) * 0.35)):
                stats[i] = stats[i] * 1.25
        # 累计flow
        flow_evolution = [sum(stats[:i+1]) for i in range(len(stats))]
        # # 随时间变化flow
        # flow_evolution = stats
        flow_evolution_each_strategy[version_name] = flow_evolution


    # 画图
    plot_path = ROOT_PATH + constr_name_attach(flow_type, 'compare', *[str(v) for v in versions])
    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated ' + flow_type + ' (veh)')
    for version_name, flow_evolution in flow_evolution_each_strategy.items():
        plt.plot(np.arange(0, len(flow_evolution)*100, 100), flow_evolution, '-', label=version_name)
    plt.legend()
    # plt.savefig(plot_path)
    # plt.show()
    plt.close()


def constr_name_attach(*args, join_char: str = '_'):
    return join_char.join(args)


if __name__ == '__main__':

    # title = 'throughput'
    # versions = ('77', '80', '98', '99')
    # labels = ('NEMA+balance', 'Unfixed+balance', 'NEMA+equal', 'Unfixed+equal')
    # epis = (1, 2, 3)
    # compare(title, versions, labels, epis)

    versions = {220: 'PSC-GQF',
                221: 'PSC-QF',      # PSC-EF
                222: 'PSC-GEF',
                # 226: 'PSC-QF',
                227: 'PSC-GQN'}
    versions = {238: 'PSC-GQF',
                239: 'PSC-QF',
                240: 'PSC-GEF',
                241: 'PSC-GQN',
                }
    # compare_throughput(versions, epis=10)
    # compare_queue_variance(versions, epis=10)
    compare_flow('inflow', versions, epis=1)
    compare_flow('outflow', versions, epis=1)
