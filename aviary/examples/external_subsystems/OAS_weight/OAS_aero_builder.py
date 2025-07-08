import openmdao.api as om

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.examples.external_subsystems.OAS_weight.OAS_aero_analysis import OASAero
from aviary.variable_info.variables import Aircraft


class OASAeroBuilder(SubsystemBuilderBase):
    def __init__(self, name='aero_analysis'):
        super().__init__(name)
        
    def build_external_subsystem(self, aviary_inputs, num_nodes):
        return OASAero(
            aviary_inputs=aviary_inputs,
            num_nodes=num_nodes
        )

    # def build_mission(self, num_nodes, aviary_inputs, **kwargs):
    #     aero_group = om.Group()
    #     aero_group.add_subsystem(
    #         'liftdrag',
    #         OASAero(
    #             num_nodes=num_nodes,
    #             aviary_inputs=aviary_inputs,
    #             k_lam=0.05,
    #             CL0=0.0,
    #             CD0=0.015,
    #         ),
    #         promotes_inputs=[
    #             'v',
    #             'alpha',
    #             'altitude',
    #         ],
    #         promotes_outputs=[
    #             'CL', 'CD',
    #             'wing_perf.CL', 'wing_perf.CD',
    #             'htail_perf.CL', 'htail_perf.CD',
    #             Aircraft.Wing.THICKNESS_TO_CHORD,
    #             Aircraft.Wing.MAX_THICKNESS_LOCATION,
    #             Aircraft.Wing.SPAN,
    #             Aircraft.Wing.ROOT_CHORD,
    #             Aircraft.HorizontalTail.SPAN,
    #             Aircraft.HorizontalTail.ROOT_CHORD,
    #         ],
    #     )
    #     return aero_group
    
    # def mission_inputs(self, **kwargs):
    #     promotes = [
    #         'v',
    #         'alpha',
    #         'altitude',
    #     ]
    #     return promotes

    # def mission_outputs(self, **kwargs):
    #     promotes = [
    #         'CL', 'CD',
    #         'wing_perf.CL', 'wing_perf.CD',
    #         'htail_perf.CL', 'htail_perf.CD',
    #     ]
    #     return promotes