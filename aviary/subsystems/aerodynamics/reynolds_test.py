import numpy as np
import openmdao.api as om

from ambiance import Atmosphere

from aviary.examples.external_subsystems.OAS_weight.OAS_aero_analysis import ReynoldsNumber

prob = om.Problem()
nn = 1

prob.model.add_subsystem(
    'reynolds_calc',
    ReynoldsNumber(num_nodes=nn),
    promotes=['*']
)

prob.setup()

altitudes = 400
velocities = 90
length = 1.5

prob.set_val('altitude', altitudes, units='ft')
prob.set_val('velocity', velocities, units='ft/s')
prob.set_val('aircraft:wing:characteristic_length', length, units='m')

prob.run_model()

print("Temperature (K):", prob.get_val('temperature'))
print("Density (kg/m^3):", prob.get_val('density'))
print("Dynamic viscosity (kg/m·s):", prob.get_val('dynamic_viscosity'))
print("Kinematic viscosity (m^2/s):", prob.get_val('kinematic_viscosity'))
print("Reynolds number:", prob.get_val('Reynolds_number'))
