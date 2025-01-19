import numpy as np
import sumolib
import subprocess
from utils.utilize import config
import random
import xml.etree.ElementTree as ET
from utils.result_processing import plot_demand_from_turn

class TrafficGeneratorFromTurn:

    def __init__(self):
        self.turn_file_name = config['turnfile_dir']
        self.turn_probability = {}
        self.edge_flow_for_each_type = []      # each element corresponds to a demand type

    def generate_turn_probability(self, config, netdata):
        """
        Generate turn.xml for each node given the config

        Args:
            config (dict): Configurations
            netdata (dict): Dict of network data

        Return:
            None
        """
        for node, node_info in netdata['node'].items():
            if node[0] == 'O':  # outbound node
                continue
            for edge in node_info['incoming']:
                edge_turns = {}
                lanes = netdata['edge'][edge]['lanes']
                for lane in lanes:
                    lane_outgoing_edges: dict = netdata['lane'][lane]['outgoing']
                    for outgoing_lane, outgoing_info in lane_outgoing_edges.items():
                        outgoing_edge = outgoing_lane.split('_')[0]
                        outgoing_turn = outgoing_info['dir']
                        if outgoing_turn not in edge_turns:
                            edge_turns[outgoing_turn] = outgoing_edge
                for turn, to_edge in edge_turns.items():
                    if turn == 't':
                        continue    # skip the U-turn
                    if node[0] == 'I':  # internal node
                        turn_prob = config['TurnRatio']['NodePN'][turn]
                    elif node[0] == 'P':    # perimeter node
                        turn_prob = config['TurnRatio']['NodePeri'][turn]
                    else:
                        raise ValueError(f"Node type not recognized: {node}")
                    self.turn_probability[(edge, to_edge)] = turn_prob

    def generate_turn_file(self, config):
        """
        Generate turn.xml file given the turn probability and sink edges
        """
        sink_edges = config['Edge_PN_out']
        turn_filename = config['turnfile_dir']
        with open(turn_filename, "w") as turns:
            print('<turns>\n\t<edgeRelations>', file=turns)
            print(f'\t\t<interval begin="0" end="{config["max_steps"]}">', file=turns)
            for (edge, to_edge), prob in self.turn_probability.items():
                print(f'\t\t\t<edgeRelation from="{edge}" to="{to_edge}" probability="{prob}" />', file=turns)
            print(f'\t\t\t<sinks edges="{" ".join([str(edge) for edge in sink_edges])}" />', file=turns)
            print('\t\t</interval>\n\t</edgeRelations>\n</turns>', file=turns)

    def generate_flow(self, config):
        """
        Generate different types of given the config
        """
        plot_demand_from_turn(config)
        for demand_info in config['DemandConfig']:
            flow_dict = {edge: [] for edge in demand_info['FromEdges']}
            for i in range(config['Demand_interval_number']):
                edge_flow = flow_allocation(demand_info['VolumeProfile'][i] * demand_info['multiplier'],
                                            demand_info['FromEdges'], scale=config['scale'])
                for edge, flow in edge_flow.items():
                    flow_dict[edge].append(flow)
            self.edge_flow_for_each_type.append(flow_dict)

    def generate_flow_file(self, config):
        """
        Write the flow to the route file. Each route file corresponds to a demand type.
        """
        for i, demandtype in enumerate(config['DemandConfig']):
            flow_filename = config['singletype_flowfile_dir'] + f'{i+1}.flows.xml'
            with open(flow_filename, "w") as flow_file:
                print('<routes>', file=flow_file)
                begin_time, end_time = 0, 0
                for interval_cnt, interval in enumerate(config['Demand_interval']):
                    end_time += interval
                    for edge in self.edge_flow_for_each_type[i]:
                        flow = self.edge_flow_for_each_type[i][edge][interval_cnt]
                        print(f'\t<flow id="F{edge}_{interval_cnt}" from="{edge}" begin="{begin_time}" end="{end_time}" vehsPerHour="{flow}"/>', file=flow_file)
                    begin_time = end_time
                print('</routes>', file=flow_file)

    def generate_trip_file(self, config):
        """
        Generate trip file for each demand type and then combine them.

        Args:
            config (dict): Configurations

        Return:
            None
        """

        for i in range(len(config['DemandConfig'])):

            p = subprocess.run(
                [
                    "jtrrouter",
                    "--route-files",
                    config['singletype_flowfile_dir'] + f'{i+1}.flows.xml',
                    "--turn-ratio-files",
                    config['turnfile_dir'],
                    "--net-file",
                    config['netfile_dir'],
                    "--output-file",
                    config['singletype_tripfile_dir'] + f'{i+1}.trips.xml',
                    "--max-edges-factor",
                    str(config['max_edges_factor']),
                    "--randomize-flows",
                    "--remove-loops",
                    "--repair",
                    "--accept-all-destinations",
                    # disable console output
                    "--no-step-log",
                ]
            )

            ret_code = p.returncode
            if ret_code:
                raise RuntimeError(
                    f"Error occurs when calling `jtrrouter` for demand type {i}: "
                    f"error code {ret_code}."
                )

        # combine all trip files and sort by depart time
        tripfile_roots = []
        for i in range(len(config['DemandConfig'])):
            tripfile_roots.append(ET.parse(config['singletype_tripfile_dir'] + f'{i+1}.trips.xml').getroot())
        all_trips = sum([list(tripfile_root) for tripfile_root in tripfile_roots], [])
        all_trips.sort(key=lambda x: float(x.attrib['depart']))

        # write the combined trip file
        all_trips_root = ET.Element('routes')
        for trip in all_trips:
            all_trips_root.append(trip)
        ET.ElementTree(all_trips_root).write(config['routefile_dir'], encoding='utf-8', xml_declaration=True)
        


def flow_allocation(total_flow: int, edge_list: list, scale: float = 1.):
    """
    Allocate flow to each edge given the total flow

    Args:
        total_flow (int): Total flow
        edge_list (list): List of edges
        scale (float): Alpha value for normal distribution

    Return:
        flow_dict (dict): Dict of flow allocation
    """
    np.random.seed(47)
    while True:
        random_value = np.random.normal(0, scale, len(edge_list))
        split_ratio = [1 / len(edge_list) + r for r in random_value]
        if all([r >= 0 for r in split_ratio]):
            break
    split_ratio = split_ratio / sum(split_ratio)
    flow_dict = {edge: int(total_flow * split_ratio[i]) for i, edge in enumerate(edge_list)}
    return flow_dict
