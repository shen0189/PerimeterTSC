import matplotlib.pyplot as plt
import numpy as np
from sumolib import checkBinary
import os
import sys
import pickle
import xml.etree.cElementTree as ET
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import sys

sys.path.append("../network")


def import_train_static_configuration(config):
    """
    Add the static config to the changeable config file
    """

    # [network_config]
    if config['network'] == 'Grid':
        config['Node'] = {
            'NodePN': [6, 7, 8, 11, 12, 13, 16, 17, 18],
            'NodePeri': [25, 26, 27, 28]
        }
        config['Edge'] = list(range(0, 88))
        config['Edge_Peri'] = list(range(80, 88))
        config['Edge_PN'] = sorted([
            18, 19, 21, 22, 23, 25, 27, 34, 36, 37, 39, 38, 40, 41, 43, 42, 45, 52,
            54, 57, 56, 58, 60, 61
        ])
        config['Edge_Peri_out'] = sorted([16, 17, 20, 24, 26, 44, 62, 63, 59, 55, 53, 35])  # the outflows of PN
    elif config['network'] == 'FullGrid':
        config['Node'] = {
            'NodePN': ['I22', 'I32', 'I42', 'I23', 'I33', 'I43', 'I24', 'I34', 'I44'],
            'NodePeri': ['P11', 'P12', 'P13', 'P14', 'P21', 'P31', 'P41', 'P51',
                         'P15', 'P25', 'P35', 'P45', 'P52', 'P53', 'P54', 'P55'],
            'NodeCorner': ['P11', 'P15', 'P51', 'P55'],     # GridBufferFull1定义下使用
            'NodeOutside': ['O01', 'O02', 'O03', 'O04', 'O05',
                            'O61', 'O62', 'O63', 'O64', 'O65',
                            'O10', 'O20', 'O30', 'O40', 'O50',
                            'O16', 'O26', 'O36', 'O46', 'O56']
        }

        config['Node_PN'] = ['I22', 'I32', 'I42', 'I23', 'I33', 'I43', 'I24', 'I34', 'I44']
        config['Node_Peri'] = ['P11', 'P12', 'P13', 'P14', 'P21', 'P31', 'P41', 'P51',
                               'P15', 'P25', 'P35', 'P45', 'P52', 'P53', 'P54', 'P55'],
        config['Node_Corner'] = ['P11', 'P15', 'P51', 'P55'],
        config['Node_outside'] = ['O01', 'O02', 'O03', 'O04', 'O05', 'O61', 'O62', 'O63', 'O64', 'O65',
                                  'O10', 'O20', 'O30', 'O40', 'O50', 'O16', 'O26', 'O36', 'O46', 'O56']
        config['peri_node_types'] = ['Node_Peri_high', 'Node_Peri_low', 'Node_Peri_mixed']
        config['Node_Peri_high'] = ['P12', 'P13', 'P14', 'P15', 'P25', 'P35', 'P45']
        config['Node_Peri_low'] = ['P21', 'P31', 'P41', 'P51', 'P52', 'P53', 'P54']
        config['Node_Peri_mixed'] = ['P11', 'P55']

        config['Edge'] = [edge for i in range(10) for edge in range(i * 10 + 1, i * 10 + 9)] + \
                         [edge for i in range(10) for edge in range(i * 10 + 101, i * 10 + 105)]
        config['Edge_PN'] = sorted([edge for i in range(10) for edge in range(i * 10 + 1, i * 10 + 9)])
        config['Edge_PN_inside'] = [edge for i in range(2, 8) for edge in range(i * 10 + 1, i * 10 + 9)]
        config['Edge_PN_exit'] = sorted([32, 52, 72, 28, 48, 68, 37, 57, 77, 21, 41, 61])  # 从3*3到5*5的路段
        config['Edge_Peri'] = sorted(
            [edge for i in range(0, 10, 2) for edge in [i * 10 + 102, i * 10 + 114, i * 10 + 103, i * 10 + 111]])
        config['Edge_Peri_out'] = sorted(
            [edge for i in range(0, 10, 2) for edge in [i * 10 + 104, i * 10 + 112, i * 10 + 101, i * 10 + 113]])
        config['peri_inflow_edge_type'] = ['Edge_Peri_north_in', 'Edge_Peri_west_in',
                                           'Edge_Peri_south_in', 'Edge_Peri_east_in']
        config['Edge_Peri_north_in'] = sorted([edge for i in range(0, 10, 2) for edge in [i * 10 + 103]])
        config['Edge_Peri_west_in'] = sorted([edge for i in range(0, 10, 2) for edge in [i * 10 + 102]])
        config['Edge_Peri_south_in'] = sorted([edge for i in range(0, 10, 2) for edge in [i * 10 + 111]])
        config['Edge_Peri_east_in'] = sorted([edge for i in range(0, 10, 2) for edge in [i * 10 + 114]])
        config['Edge_Peri_high_demand'] = sorted(config['Edge_Peri_north_in'] + config['Edge_Peri_west_in'])
        config['Edge_Peri_low_demand'] = sorted(config['Edge_Peri_south_in'] + config['Edge_Peri_east_in'])
        config['peri_outflow_edge_type'] = ['Edge_Peri_north_out', 'Edge_Peri_west_out',
                                            'Edge_Peri_south_out', 'Edge_Peri_east_out']
        config['Edge_Peri_north_out'] = sorted([edge for i in range(0, 10, 2) for edge in [i * 10 + 113]])
        config['Edge_Peri_west_out'] = sorted([edge for i in range(0, 10, 2) for edge in [i * 10 + 112]])
        config['Edge_Peri_south_out'] = sorted([edge for i in range(0, 10, 2) for edge in [i * 10 + 101]])
        config['Edge_Peri_east_out'] = sorted([edge for i in range(0, 10, 2) for edge in [i * 10 + 104]])

    elif config['network'] == 'Bloomsbury':
        config['Node'] = {
            'NodePN': [13, 12, 22, 21, 20, 19, 18, 17, 34],
            'NodePeri': [50, 51, 52, 53, 54, 55]
        }
        # read edge data
        df = pd.read_csv("network\Bloomsbury\edge_bloom.csv", sep=';', usecols=['edge_id', 'edge_PN'])

        # config['Edge'] = sorted(df['edge_id'].to_numpy)
        config['Edge'] = sorted(df['edge_id'].values.tolist())
        config['Edge_PN'] = sorted(df[df['edge_PN'] == 1]['edge_id'].values.tolist())
        config['Edge_Peri'] = sorted(df[df['edge_PN'] == 0]['edge_id'].values.tolist())

    # [turn ratio config]
    if config['network'] == 'FullGrid':
        if config['network_version'] in ['GridBufferFull', 'GridBufferFull2']:
            config['TurnRatio'] = {
                'EdgePN': {'l': 0.3, 's': 0.5, 'r': 0.2},
                'EdgePeri': {'l': 0.3, 's': 0.5, 'r': 0.2},
            }
        elif config['network_version'] == 'GridBufferFull1':
            config['TurnRatio'] = {
                'EdgePN': {'l': 0.3, 's': 0.5, 'r': 0.2},
                'EdgePeri': {'l': 0.3, 's': 0.5, 'r': 0.2},
                'EdgeCorner': {'l': 0.3, 's': 0.7, 'r': 0.3}
            }
        config['max_edges_factor'] = 0.8
    else:
        raise NotImplementedError

    # [perimeter signal config]
    if config['network'] == 'Grid':  # 'Grid':
        # config['EdgeCross'] = np.array([92, 93, 94, 95])
        config['Peri_info'] = {
            'A2': {
                'node': '2',
                'edge': 80,
                'in_edges': [8, 20, 3, 80],  # 按照sumo中定义phase state的顺序 从北进口开始顺时针方向
                'down_edge': 7,
                'buffer_edges': [5, 6, 7],
                'external_in_edges': [80],
                'external_out_edges': [81],
                'internal_in_edges': [8, 20, 3],
                'internal_out_edges': [5, 6, 7],
                'inflow_lane_groups': [3],   # 列表长度代表车道组数量, 第i个元素代表车道组i的车道数量, 从最外侧开始数
                'nema_plan': {'ring1': ['3_s', '8_l', '', '20_s'],
                              'ring2': ['8_s', '3_l', '', '80_s', '20_l']},
                'phase_info':
                    ['', 'yellow_phase', '', 'yellow_phase', 'control_phase', 'yellow_phase', 'control_phase',
                     'yellow_phase'],  # 根据group-based优化结果修改
                'slot_stage_plan': {'stage1': ('3_l', '3_s'), 'stage2': ('3_l', '8_l'),
                                    'stage3': ('8_l', '8_s'), 'stage4': ('3_s', '8_s'),
                                    'stage5': ('20_l',), 'stage6': ('80_s',), 'stage7': ('20_s', '20_l'),
                                    'stage8': ('20_s', '80_s')}
            },
            'C4': {
                'node': '14',
                'edge': 82,
                'in_edges': [82, 64, 44, 30],
                'down_edge': 47,
                'buffer_edges': [48, 47, 46],
                'external_in_edges': [82],
                'external_out_edges': [83],
                'internal_in_edges': [30, 44, 64],
                'internal_out_edges': [48, 47, 46],
                'inflow_lane_groups': [3],
                'nema_plan': {'ring1': ['82_s', '44_l', '', '64_s', '30_l'],
                              'ring2': ['44_s', '', '30_s', '64_l']},
                'phase_info': ['control_phase', 'yellow_phase', 'control_phase', 'yellow_phase',
                               '', 'yellow_phase', '', 'yellow_phase'],
                'slot_stage_plan': {'stage1': ('30_l', '30_s'), 'stage2': ('30_l', '64_l'),
                                    'stage3': ('64_l', '64_s'), 'stage4': ('30_s', '64_s'),
                                    'stage5': ('44_l',), 'stage6': ('82_s',), 'stage7': ('44_s', '44_l'),
                                    'stage8': ('44_s', '82_s')}
            },
            'E2': {
                'node': '22',
                'edge': 84,
                'in_edges': [76, 84, 71, 59],
                'down_edge': 72,
                'buffer_edges': [73, 72, 74],
                'external_in_edges': [84],
                'external_out_edges': [85],
                'internal_in_edges': [59, 71, 76],
                'internal_out_edges': [73, 72, 74],
                'inflow_lane_groups': [3],
                'nema_plan': {'ring1': ['76_s', '71_l', '', '84_s', '59_l'],
                              'ring2': ['71_s', '76_l', '', '59_s']},
                'phase_info': ['', 'yellow_phase', '', 'yellow_phase',
                               'control_phase', 'yellow_phase', 'control_phase', 'yellow_phase'],
                'slot_stage_plan': {'stage1': ('71_l', '71_s'), 'stage2': ('71_l', '76_l'),
                                    'stage3': ('76_l', '76_s'), 'stage4': ('71_s', '76_s'),
                                    'stage5': ('59_l',), 'stage6': ('84_s',), 'stage7': ('59_s', '59_l'),
                                    'stage8': ('59_s', '84_s')}
            },
            'C0': {
                'node': '10',
                'edge': 86,
                'in_edges': [35, 49, 86, 15],
                'down_edge': 32,
                'buffer_edges': [31, 32, 33],
                'external_in_edges': [86],
                'external_out_edges': [87],
                'internal_in_edges': [15, 35, 49],
                'internal_out_edges': [31, 32, 33],
                'inflow_lane_groups': [3],
                'nema_plan': {'ring1': ['35_s', '', '49_s', '15_l'],
                              'ring2': ['86_s', '35_l', '', '15_s', '49_l']},
                'phase_info': ['control_phase', 'yellow_phase', 'control_phase', 'yellow_phase',
                               '', 'yellow_phase', '', 'yellow_phase'],
                'slot_stage_plan': {'stage1': ('15_l', '15_s'), 'stage2': ('15_l', '49_l'),
                                    'stage3': ('49_l', '49_s'), 'stage4': ('15_s', '49_s'),
                                    'stage5': ('35_l',), 'stage6': ('86_s',), 'stage7': ('35_s', '35_l'),
                                    'stage8': ('35_s', '86_s')}
            }
        }
    elif config['network'] == 'FullGrid':
        config['Peri_info'] = {
            'P11': {'node': 'P11', 'gated_edge': [102, 111]},
            'P12': {'node': 'P12', 'gated_edge': [122]},
            'P13': {'node': 'P13', 'gated_edge': [142]},
            'P14': {'node': 'P14', 'gated_edge': [162]},
            'P21': {'node': 'P21', 'gated_edge': [131]},
            'P31': {'node': 'P31', 'gated_edge': [151]},
            'P41': {'node': 'P41', 'gated_edge': [171]},
            'P51': {'node': 'P51', 'gated_edge': [191, 114]},
            'P15': {'node': 'P15', 'gated_edge': [182, 103]},
            'P25': {'node': 'P25', 'gated_edge': [123]},
            'P35': {'node': 'P35', 'gated_edge': [143]},
            'P45': {'node': 'P45', 'gated_edge': [163]},
            'P52': {'node': 'P52', 'gated_edge': [134]},
            'P53': {'node': 'P53', 'gated_edge': [154]},
            'P54': {'node': 'P54', 'gated_edge': [174]},
            'P55': {'node': 'P55', 'gated_edge': [194, 183]}
        }
        config['lane_group_info'] = {
            1: ('east', 'l'),
            2: ('west', 's'),
            3: ('south', 'l'),
            4: ('north', 's'),
            5: ('west', 'l'),
            6: ('east', 's'),
            7: ('north', 'l'),
            8: ('south', 's')
        }
        config['phase_sequence'] = {
            'EW-NS-HL-HL': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            'EW-NS-HH-HL': [[[1, 2], [3, 4]], [[6, 5], [7, 8]]],
            'EW-NS-HL-HH': [[[1, 2], [3, 4]], [[5, 6], [8, 7]]],
            'EW-NS-HH-HH': [[[1, 2], [3, 4]], [[6, 5], [8, 7]]],
            'NS-EW-HL-HL': [[[3, 4], [1, 2]], [[7, 8], [5, 6]]],
            'NS-EW-HH-HL': [[[3, 4], [1, 2]], [[8, 7], [5, 6]]],
            'NS-EW-HL-HH': [[[3, 4], [1, 2]], [[7, 8], [6, 5]]],
            'NS-EW-HH-HH': [[[3, 4], [1, 2]], [[8, 7], [6, 5]]]
        }
        config['phase_sequence_corner_type1'] = {
            'EW-NS-HL-HL': [[[1, 2], [4]], [[5, 6], [8]]],
            'EW-NS-HH-HL': [[[1, 2], [4]], [[6, 5], [8]]],
            'NS-EW-HL-HL': [[[4], [1, 2]], [[8], [5, 6]]],
            'NS-EW-HL-HH': [[[4], [1, 2]], [[8], [6, 5]]],
        }       # P11, P55
        config['phase_sequence_corner_type2'] = {
            'EW-NS-HL-HL': [[[2], [3, 4]], [[6], [7, 8]]],
            'EW-NS-HL-HH': [[[2], [3, 4]], [[6], [8, 7]]],
            'NS-EW-HH-HL': [[[3, 4], [2]], [[8, 7], [6]]],
            'NS-EW-HL-HH': [[[3, 4], [2]], [[7, 8], [6]]],
        }       # P15, P51
    elif config['network'] == 'Bloomsbury':
        config['Peri_info'] = {
            '15': {
                'edge': 1201,
                'down_edge': 1050,
                'buffer_edges': [1050, 1046, 1048],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '3': {
                'edge': 1202,
                'down_edge': 1010,
                'buffer_edges': [1010, 1036, 1035, 1034],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '8': {
                'edge': 1205,
                'down_edge': 1023,
                'buffer_edges': [1023, 1027, 1028],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '32': {
                'edge': 1207,
                'down_edge': 1113,
                'buffer_edges': [1113, 1117, 1115],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '43': {
                'edge': 1209,
                'down_edge': 1150,
                'buffer_edges': [1150, 1123, 1124, 1125],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '39': {
                'edge': 1211,
                'down_edge': 1140,
                'buffer_edges': [1136, 1135, 1136],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
        }

    # [dir]
    config['savefile_dir'] = "output/"
    config['models_path_name'] = "output/model"
    config['plots_path_name'] = 'output/plots'
    config['cache_path_name'] = 'output/cache'
    config['stats_path_name'] = 'output/statistics'

    if config['network'] == 'Grid':
        # config['sumocfg_file_name'] = 'network/GridBuffer/GridBuffer.sumocfg'
        # config['edgefile_dir'] = "network/GridBuffer/GridBuffer.edg.xml"
        # config['netfile_dir'] = "network/GridBuffer/GridBuffer.net.xml"
        # config['routefile_dir'] = "network/GridBuffer/GridBuffer.rou.xml"
        # config['edge_outputfile_dir'] = "measurements/GridBuffer/EdgeMesurements.xml"
        # config['lane_outputfile_dir'] = "measurements/GridBuffer/EdgeMesurements_lower.xml"
        # config['queuefile_dir'] = "measurements/GridBuffer/queue.xml"

        config['sumocfg_file_name'] = 'network/GridBufferNew/GridBuffer.sumocfg'
        config['edgefile_dir'] = "network/GridBufferNew/GridBuffer.edg.xml"
        config['netfile_dir'] = "network/GridBufferNew/GridBuffer.net.xml"
        config['routefile_dir'] = "network/GridBufferNew/GridBuffer.rou.xml"
        config['edge_outputfile_dir'] = "measurements/GridBufferNew/EdgeMesurements.xml"
        config['lane_outputfile_dir'] = "measurements/GridBufferNew/EdgeMesurements_lower.xml"
        config['queuefile_dir'] = "measurements/GridBufferNew/queue.xml"
        config['tripfile_dir'] = "measurements/GridBufferNew/trip.xml"

        # tls configure
        if config['lower_mode'] == 'OAM':  # 'FixTime'  #'OAM' # 'MaxPressure'
            config['tls_config_name'] = './code/tls/tls_new1.pkl'
        elif config['lower_mode'] == 'MaxPressure':
            config['tls_config_name'] = './code/tls/tls_new07.pkl'

    if config['network'] == 'FullGrid':
        config['sumocfg_file_name'] = '/'.join(('network', config['network_version'], 'GridBuffer.sumocfg'))
        config['edgefile_dir'] = '/'.join(('network', config['network_version'], 'GridBuffer.edg.xml'))
        config['netfile_dir'] = '/'.join(('network', config['network_version'], 'GridBuffer.net.xml'))
        # config['routefile_dir'] = '/'.join(('network', config['network_version'], 'GridBuffer.rou.xml'))
        config['singletype_flowfile_dir'] = '/'.join(('network', config['network_version'], 'Demands', 'DemandType'))
        config['turnfile_dir'] = '/'.join(('network', config['network_version'], 'Demands', 'GridBuffer.turn.xml'))
        config['singletype_tripfile_dir'] = '/'.join(('network', config['network_version'], 'Demands', 'DemandType'))
        config['routefile_dir'] = '/'.join(('network', config['network_version'], 'GridBuffer.rou.xml'))
        config['edge_outputfile_dir'] = '/'.join(('measurements', config['network_version'], 'EdgeMeasurements.xml'))
        config['lane_outputfile_dir'] = '/'.join(('measurements', config['network_version'], 'EdgeMeasurements_lower.xml'))
        config['queuefile_dir'] = '/'.join(('measurements', config['network_version'], 'queue.xml'))
        config['tripfile_dir'] = '/'.join(('measurements', config['network_version'], 'trip.xml'))

        # tls configure
        if config['upper_mode'] == 'Static':
            config['normal_plan'] = {
                'GGGrrrGGGrrr': 25,
                'yyyrrryyyrrr': 5,
                'rrGrrrrrGrrr': 15,
                'rryrrrrryrrr': 5,
                'rrrGGGrrrGGG': 25,
                'rrryyyrrryyy': 5,
                'rrrrrGrrrrrG': 15,
                'rrrrryrrrrry': 5
            }
        if config['lower_mode'] == 'MaxPressure':
            if config['network_version'] == 'GridBufferFull':
                config['tls_config_name'] = './code/tls/tls_full_0.pkl'
            elif config['network_version'] == 'GridBufferFull1':
                config['tls_config_name'] = './code/tls/tls_full_1.pkl'
            elif config['network_version'] == 'GridBufferFull2':
                config['tls_config_name'] = './code/tls/tls_full_2.pkl'
        # else:
        #     raise NotImplementedError
    
    if config['network'] == 'Bloomsbury':
        config['sumocfg_file_name'] = 'network/Bloomsbury/Bloomsbury.sumocfg'
        config['netfile_dir'] = "network/Bloomsbury/Bloomsbury.net.xml"
        config['edgefile_dir'] = "network/Bloomsbury/Bloomsbury.edg.xml"
        config['routefile_dir'] = "network/Bloomsbury/Bloomsbury.rou.xml"
        config['queuefile_dir'] = "measurements/Bloomsbury/queue.xml"
        config['edge_outputfile_dir'] = "measurements/Bloomsbury/EdgeMesurements.xml"
        config['lane_outputfile_dir'] = "measurements/Bloomsbury/EdgeMesurements_lower.xml"

        # tls configure
        if config['lower_mode'] == 'OAM':  # 'FixTime'  #'OAM' # 'MaxPressure'
            config['tls_config_name'] = './code/tls/tls_bloom_1.pkl'
        elif config['lower_mode'] == 'MaxPressure':
            config['tls_config_name'] = './code/tls/tls_bloom_05.pkl'

    return config


def get_NodeConfig(edgefile_dir):
    ''' read the edge file 'edg.xml', obtain the indege and outedge of each node
    '''
    tree = ET.ElementTree(file=edgefile_dir)
    root = tree.getroot()
    NodeConfig = {}
    for child in root:
        edge_id = int(child.attrib['id'])
        From_Node = int(child.attrib['from'])
        To_Node = int(child.attrib['to'])

        # add From_Node
        if From_Node not in NodeConfig:
            NodeConfig[From_Node] = {'InEdge': [], 'OutEdge': []}

        NodeConfig[From_Node]['OutEdge'].append(edge_id)

        # add From_Node
        if To_Node not in NodeConfig:
            NodeConfig[To_Node] = {'InEdge': [], 'OutEdge': []}

        NodeConfig[To_Node]['InEdge'].append(edge_id)

    return NodeConfig


# class Test:
#     def __init__(self):
#         self.obj_list = []
#         self.reward_list = []
#         self.penalty_list = []
#         self.accu_list = []
#         self.throu_list = []
#         self.action_list = []
#
#     def record_data(self, cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, actions):
#         self.obj_list.append(cumul_obj)
#         self.reward_list.append(cumul_reward)
#         self.penalty_list.append(cumul_penalty)
#         self.accu_list.append(accu_episode)
#         self.throu_list.append(throughput_episode)
#         self.action_list.append(actions)
#
#     def save_data_test(self):
#         ''' save objective, reward, penalty, throughput, actions along the testing process
#         '''
#         data_test = {}
#
#         data_test['obj_list'] = self.obj_list
#         data_test['reward_list'] = self.reward_list
#         data_test['penalty_epis'] = self.penalty_list
#         data_test['accu_list'] = self.accu_list
#         data_test['throu_list'] = self.throu_list
#         data_test['action_list'] = self.action_list
#
#         save_dir = config['models_path_name'] + 'data_test.pkl'
#         with open(save_dir, 'wb') as f:
#             pickle.dump([data_test], f)
#
#         print('###### Data save: Success ######')



