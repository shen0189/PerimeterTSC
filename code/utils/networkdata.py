import pickle
import sys, subprocess, os
from copy import copy
import inspect
import sumolib
import traci
from utils.trafficsignalcontroller import TrafficSignalController
from utils.utilize import config


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
'''
# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools")) # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools")) # tutorial in docs
    #from sumolib import checkBinary
    import sumolib
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
'''

import numpy as np

class NetworkData:
    def __init__(self, net_fp, sumo_cmd):
        print(net_fp) 
        self.net = sumolib.net.readNet(net_fp)
        ###get edge data
        self.node_data, self.tls_data = self._get_node_data(self.net)
        self.edge_data = self._get_edge_data(self.net)
        self.lane_data = self._get_lane_data(self.net, self.edge_data)
        self.sumo_cmd = sumo_cmd

        print("SUCCESSFULLY GENERATED NET DATA")

    def get_net_data(self):
        ''' generate network data dict with lane, edge,node,tls etc'''
        self.netdata = {'lane':self.lane_data, \
                        'edge':self.edge_data, \
                        'origin':self._find_origin_edges(), \
                        'destination':self._find_destination_edges(), \
                        'node':self.node_data, 
                        'tls':self.tls_data}

        return self.netdata

    ### get OD edges
    def _find_destination_edges(self):
        next_edges = { e:0 for e in self.edge_data }
        for e in self.edge_data:
            for next_e in self.edge_data[e]['incoming']:
                next_edges[next_e] += 1
                                                                 
        destinations = [ e for e in next_edges if next_edges[e] == 0]
        return destinations

    def _find_origin_edges(self):
        next_edges = { e:0 for e in self.edge_data }
        for e in self.edge_data:
            for next_e in self.edge_data[e]['outgoing']:
                next_edges[next_e] += 1

        origins = [ e for e in next_edges if next_edges[e] == 0]
        return origins

    ### helper functions: read the network
    def _get_edge_data(self, net):
        edges = net.getEdges()
        edge_data = {str(edge.getID()):{} for edge in edges}
        PN_length = 0

        for edge in edges:
            edge_ID = str(edge.getID())
            edge_data[edge_ID]['lanes'] = [str(lane.getID()) for lane in edge.getLanes()]
            edge_data[edge_ID]['length'] = float(edge.getLength())
            edge_data[edge_ID]['outgoing'] = [str(out.getID()) for out in edge.getOutgoing()]
            edge_data[edge_ID]['noutgoing'] = len(edge_data[edge_ID]['outgoing'])
            edge_data[edge_ID]['nlanes'] = len(edge_data[edge_ID]['lanes'])
            edge_data[edge_ID]['incoming'] = [str(inc.getID()) for inc in edge.getIncoming()]
            edge_data[edge_ID]['outnode'] = str(edge.getFromNode().getID())
            edge_data[edge_ID]['incnode'] = str(edge.getToNode().getID())
            edge_data[edge_ID]['speed'] = float(edge.getSpeed())
          
            ###coords for each edge
            incnode_coord = edge.getFromNode().getCoord()
            outnode_coord = edge.getToNode().getCoord()
            edge_data[edge_ID]['coord'] = np.array([incnode_coord[0], incnode_coord[1], outnode_coord[0], outnode_coord[1]]).reshape(2,2)
            #print edge_data[edge_ID]['coord']

            if int(edge_ID) in config['Edge_PN']:
                PN_length += edge_data[edge_ID]['length'] * edge_data[edge_ID]['nlanes'] / 1000
        config['PN_total_length'] = PN_length

        return edge_data 

    def _get_lane_data(self, net, edge_data):
        '''create lane dict from lane_ids '''
        lane_ids = []
        for edge in self.edge_data:
            lane_ids.extend(self.edge_data[edge]['lanes'])

        lanes = [net.getLane(lane) for lane in lane_ids]
        #lane data dict
        lane_data = {lane:{} for lane in lane_ids}

        for lane in lanes:
            # print(lane)
            lane_id = lane.getID()
            lane_data[ lane_id ]['length'] = lane.getLength()
            lane_data[lane_id]['speed'] = lane.getSpeed()
            
            ''' get the edge it belongs '''
            edge = str(lane.getEdge().getID())
            lane_data[lane_id]['edge'] = edge
            
            ''' get the ToNode it forwards to'''
            lane_data[lane_id]['ToNode'] = edge_data[edge]['incnode']


            
            ''' get outgoing lanes'''
            ##.getOutgoing() returns a Connection type, which we use to determine the next lane
            lane_data[ lane_id ]['outgoing'] = {}
            moveid = []
            for conn in lane.getOutgoing():
                out_id = str(conn.getToLane().getID())
                lane_data[ lane_id ]['outgoing'][out_id] = {'dir':str(conn.getDirection()), 'index':conn.getTLLinkIndex()}
                moveid.append(str(conn.getDirection()))
            lane_data[ lane_id ]['movement'] = ''.join(sorted(moveid))
            #create empty list for incoming lanes 
            lane_data[ lane_id ]['incoming'] = []
               
        ''' get incomming lanes '''
        ## determine incoming lanes using outgoing lanes data
        for lane in lane_data:
            for inc in lane_data:
                if lane == inc:
                    continue
                else:
                    if inc in lane_data[lane]['outgoing']:
                        lane_data[inc]['incoming'].append(lane)

        return lane_data

    def _get_node_data(self, net):
        '''
        1: get node data: dict
            Node is "ALL" intersections
        2: get tls data: dict
            tls is the node "WITH" traffic signals
        '''
        #read network from .netfile
        nodes = net.getNodes()
        node_data = {node.getID():{} for node in nodes}

        for node in nodes:
            node_id = node.getID()
            #get incoming/outgoing edge
            node_data[node_id]['incoming'] = set(str(edge.getID()) for edge in node.getIncoming())
            node_data[node_id]['outgoing'] = set(str(edge.getID()) for edge in node.getOutgoing())
            node_data[node_id]['tlsindex'] = { conn.getTLLinkIndex():str(conn.getFromLane().getID()) for conn in node.getConnections()}
            node_data[node_id]['tlsindexdir'] = { conn.getTLLinkIndex():str(conn.getDirection()) for conn in node.getConnections()}

            if node_id == '-13968':
                missing = []
                negative = []
                for i in range(len(node_data[node_id]['tlsindex'])):
                    if i not in node_data[node_id]['tlsindex']:
                        missing.append(i)

                for k in node_data[node_id]['tlsindex']:
                    if k < 0  :
                        negative.append(k)
              
                for m,n in zip(missing, negative):
                    node_data[node_id]['tlsindex'][m] = node_data[node_id]['tlsindex'][n]
                    del node_data[node_id]['tlsindex'][n]
                    #for index dir
                    node_data[node_id]['tlsindexdir'][m] = node_data[node_id]['tlsindexdir'][n]
                    del node_data[node_id]['tlsindexdir'][n]
            
            #get XY coords
            pos = node.getCoord()
            node_data[node_id]['x'] = pos[0]
            node_data[node_id]['y'] = pos[1]

        tls_data = {str(node):node_data[node] for node in node_data if "traffic_light" in net.getNode(node).getType()} 

        return node_data, tls_data

    ### update network
    def update_netdata(self):

        traci.start(self.sumo_cmd)
        tl_junc = self._get_traffic_lights()
        tsc = { tl_id:TrafficSignalController( tl_id, junc_id, 'train', self.netdata)  
                        for tl_id, junc_id in tl_junc.items() }

        for t_id, t_value in tsc.items():
            self.netdata['tls'][t_value.junc_id]['incoming_lanes'] = t_value.incoming_lanes
            self.netdata['tls'][t_value.junc_id]['tl_id'] = t_id
            self.netdata['node'][t_value.junc_id]['tl_id'] = t_id
            # self.netdata['tls'][t]['green_phases'] = tsc[t].green_phases


        ## set perimeter tsc    
        tsc_peri = self.update_perimeter(tsc)

        traci.close()

        return  tsc, tsc_peri

    def update_perimeter(self,tsc):
        ''' get the tsc of the perimeter
        '''
        tsc_peri=[]
        for t_id, t_value in tsc.items():
            if t_id in config['Peri_info'].keys():
                tsc_peri.append(tsc[t_id])
                config['Peri_info'][t_id]['tsc'] = t_value

                # get program
                logic = traci.trafficlight.getAllProgramLogics(t_id)[0]
                t_value.logic = logic
                

                # # get phase
                # for idx, signal in enumerate(logic.phases):
                #     if  config['Peri_info'][t_id]['phase_info'][idx] =='control_phase':
                #     # if 'y' not in signal.state and 'r' not in signal.state:
                #         t_value.green_phase = signal.state
                #         t_value.green_phase_index = idx
                #
                #     elif config['Peri_info'][t_id]['phase_info'][idx] =='':
                #         t_value.red_phase = signal.state
                #         t_value.red_phase_index = idx
        return tsc_peri


    def _get_traffic_lights(self):
        '''find all the junctions with traffic lights
            match the traffic light with the junction
        '''
        tl_junc={}
        trafficlights = traci.trafficlight.getIDList()
        inters = copy(self.netdata['tls'])
        for tl in trafficlights:
            in_lanes= traci.trafficlight.getControlledLanes(tl)
            for inter_id, inter in inters.items():
                if set(in_lanes) == set(inter['tlsindex'].values()) :
                    tl_junc[tl] = inter_id
                    inters.pop(inter_id)
                    break

        return tl_junc

