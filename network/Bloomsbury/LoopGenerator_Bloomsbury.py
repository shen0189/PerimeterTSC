# -*- coding: utf-8 -*-
'''
线圈生成
'''

import numpy as np
import pandas as pd


# add文件中的头部内容
text =  r"""<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 12/07/20 10:13:03 by Eclipse SUMO netedit Version 1.7.0
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <sumo-net-file value="E:\StevenSU\OneDrive - City University of Hong Kong\Research in CityU\Code\SUMO_MFD\Bilevel_3.0\network\GridBuffer.net.xml"/>
    </input>

    <output>
        <output-file value="E:\StevenSU\OneDrive - City University of Hong Kong\Research in CityU\Code\SUMO_MFD\Bilevel_3.0\network\GridBuffer.net.xml"/>
    </output>

    <processing>
        <geometry.min-radius.fix.railways value="false"/>
        <geometry.max-grade.fix value="false"/>
        <offset.disable-normalization value="true"/>
        <lefthand value="false"/>
    </processing>

    <junctions>
        <no-turnarounds value="true"/>
        <junctions.corner-detail value="5"/>
        <junctions.limit-turn-speed value="5.5"/>
        <rectangular-lane-cut value="false"/>
    </junctions>

    <pedestrian>
        <walkingareas value="false"/>
    </pedestrian>

    <netedit>
        <additional-files value="E:\StevenSU\OneDrive - City University of Hong Kong\Research in CityU\Code\SUMO_MFD\Bilevel_3.0\network\GridBuffer.add.xml"/>
    </netedit>

    <report>
        <aggregate-warnings value="5"/>
    </report>

</configuration>
-->"""


Edge = np.array(range(164))


"""每个出入口都生成一个线圈，子路网也是"""
with open(r"Bloomsbury.add.xml", "w") as routes:
    print(text, file=routes)    
    print('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">',\
          file=routes)

    # 每条路一个文件
    for i in range(len(Edge)):
        edgenum = i+1000
        if edgenum == 0:
            for lanenum in range(3):
                print(f'\t<e1Detector id="000{lanenum}_in" lane="{int(Edge[i])}_{lanenum}" pos="1" freq="10.00" file="1.xml"/>',file=routes)
    # for i in range(len(EdgeOut)):
    #     for lanenum in range(4):
                print(f'\t<e1Detector id="000{lanenum}_out" lane="{int(Edge[i])}_{lanenum}" pos="472.2" freq="10.00" file="1.xml"/>',file=routes)

        else:
            for lanenum in range(3):
                print(f'\t<e1Detector id="{int(edgenum*100+lanenum)}_in" lane="{int(edgenum)}_{lanenum}" pos="1" freq="50.00" file="1.xml"/>',file=routes)
        # for i in range(len(EdgeOut)):
        #     for lanenum in range(4):
                print(f'\t<e1Detector id="{int(edgenum*100+lanenum)}_out" lane="{int(edgenum)}_{lanenum}" pos="472.2" freq="50.00" file="1.xml"/>',file=routes)

    # 子路网loop
    # for i in range(len(EdgeIn_subnet)):
    #     for lanenum in range(4):
    #         print(f'\t<e1Detector id="{int(EdgeIn_subnet[i]*100+lanenum)}" lane="{int(EdgeIn_subnet[i])}_{lanenum}" pos="455" freq="30.00" file="1.xml"/>',file=routes)
    # for i in range(len(EdgeOut_subnet)):
    #     for lanenum in range(4):
    #         print(f'\t<e1Detector id="{int(EdgeOut_subnet[i]*100+lanenum)}" lane="{int(EdgeOut_subnet[i])}_{lanenum}" pos="20" freq="30.00" file="1.xml"/>',file=routes)
  
    print('</additional>', file=routes)
    