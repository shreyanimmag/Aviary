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

