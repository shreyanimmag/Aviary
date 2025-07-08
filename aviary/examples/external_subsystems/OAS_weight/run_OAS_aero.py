import numpy as np

import aviary.api as av
from aviary.examples.external_subsystems.OAS_weight.OAS_aero_builder import OASAeroBuilder
from aviary.variable_info.variables import Aircraft, Dynamic, Settings, Mission
from aviary.variable_info.enums import EquationsOfMotion, LegacyCode

aero_builder = OASAeroBuilder()

phase_info = {
    'pre_mission': {
        'include_takeoff': False,
        'optimize_mass': True,
        'num_nodes': 1,
        'external_subsystems': [],
        'constraints': [],
        'design_parameters': [],
        'objective': None,
        'user_options': {},  # add this line
    },
    'climb': {
        'num_nodes': 1,
        'external_subsystems': [],
        'constraints': [],
        'design_parameters': [],
        'objective': None,
        'user_options': {},
    },
    'cruise': {
        'num_nodes': 1,
        'external_subsystems': [aero_builder],
        'constraints': [],
        'design_parameters': [],
        'objective': None,
        'user_options': {},
    },
    'descent': {
        'num_nodes': 1,
        'external_subsystems': [],
        'constraints': [],
        'design_parameters': [],
        'objective': None,
        'user_options': {},
    },
    'landing': {
        'num_nodes': 1,
        'external_subsystems': [],
        'constraints': [],
        'design_parameters': [],
        'objective': None,
        'user_options': {},
    }
}

aviary_inputs = av.AviaryValues()

# just bc it's begging for these
aviary_inputs.set_val(Settings.EQUATIONS_OF_MOTION,EquationsOfMotion.HEIGHT_ENERGY)
aviary_inputs.set_val(Settings.MASS_METHOD, LegacyCode.FLOPS)
aviary_inputs.set_val(Settings.AERODYNAMICS_METHOD, LegacyCode.FLOPS)
aviary_inputs.set_val(Aircraft.Engine.DATA_FILE, 'turbofan_28k.deck')
aviary_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 10, units='lbf') 
aviary_inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 10, units='lbf') 
aviary_inputs.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 3.0, units='deg')

aviary_inputs.set_val(Mission.Design.RANGE, 1.50, units='nmi')# estimate 3 laps (M2) at 3000 per lap. idrk change this later
aviary_inputs.set_val(Mission.Design.GROSS_MASS, 48.5, units='lb') # UNSW M2
aviary_inputs.set_val(Mission.Summary.CRUISE_MACH, 0.09) # based on M2 mission velocity
aviary_inputs.set_val(Mission.Design.MACH, 0.09) # based on M2 mission velocity

aviary_inputs.set_val(Dynamic.Mission.ALTITUDE, 2468, units='ft') # TIMPA at 2218, cruise altitude at 250
aviary_inputs.set_val(Dynamic.Mission.VELOCITY, 101, units='ft/s') # UNSW cruise speed M2
aviary_inputs.set_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, 1.57, units='ft') # from UNSW
aviary_inputs.set_val(Aircraft.Wing.ROOT_CHORD, 1.57, units='ft') # from UNSW
aviary_inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.12) # NACA0012
aviary_inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.1369) # MH84
aviary_inputs.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.215) # MH84
aviary_inputs.set_val(Aircraft.Wing.SPAN, 3, units='ft') # from UNSW. using half span bc symmetry
aviary_inputs.set_val(Aircraft.HorizontalTail.SPAN, 1.27, units='ft') # from UNSW. using half span bc symmetry
aviary_inputs.set_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 1.02, units='ft') # from UNSW
aviary_inputs.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 1.02, units='ft') # from UNSW

max_iter = 30
optimizer = 'SLSQP'

if __name__ == '__main__':
    prob = av.AviaryProblem()

    prob.load_inputs(aviary_inputs, phase_info=phase_info)

    prob.check_and_preprocess_inputs()
    prob.add_pre_mission_systems()
    prob.add_phases()
    prob.add_design_variables()
    prob.add_objective()

prob.add_driver(optimizer=optimizer, max_iter=max_iter)

prob.setup()
prob.run_driver()

print('CL =', prob.get_val('CL'))
print('CD =', prob.get_val('CD'))

