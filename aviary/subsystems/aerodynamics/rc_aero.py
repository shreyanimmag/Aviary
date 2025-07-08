import numpy as np
import openmdao.api as om

import aviary.constants as constants
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic

grav_metric = 9.81 # m/s^2

class DynamicPressure(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):        
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Atmosphere.DENSITY, shape=nn, units='kg/m**3')
        add_aviary_input(self, Dynamic.Mission.VELOCITY, shape=nn, units='m/s')

        add_aviary_output(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')

    def setup_partials(self):
        nn = self.options['num_nodes'] 
        rows_cols = np.arange(nn)

        self.declare_partials(
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            Dynamic.Mission.VELOCITY,
            rows = rows_cols,
            cols = rows_cols,
        ) 

        self.declare_partials(
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            Dynamic.Atmosphere.DENSITY,
            rows = rows_cols,
            cols = rows_cols,
        )

    def compute_partials(self, inputs, partials):
        V = inputs[Dynamic.Mission.VELOCITY]
        rho = inputs[Dynamic.Atmosphere.DENSITY]

        partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Mission.VELOCITY] = rho * V
        partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.DENSITY] = 0.5 * V**2


class LiftComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):        
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')
        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')
        
        self.add_input(
            name='CL', val=np.ones(nn), units='unitless'
        )

        add_aviary_output(self, Dynamic.Vehicle.LIFT, shape=nn, units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(
            Dynamic.Vehicle.LIFT,
            [Dynamic.Atmosphere.DYNAMIC_PRESSURE, 'CL'],
            rows = rows_cols,
            cols = rows_cols,
        )
        
    def compute(self, inputs, outputs):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        CL = inputs['CL']

        outputs[Dynamic.Vehicle.LIFT] = q * S * CL
    
    def compute_partials(self, inputs, partials):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        CL = inputs['CL']

        partials[Dynamic.Vehicle.LIFT, Aircraft.Wing.AREA] = q * CL
        partials[Dynamic.Vehicle.LIFT, Dynamic.Atmosphere.DYNAMIC_PRESSURE] = S * CL
        partials[Dynamic.Vehicle.LIFT, 'CL'] = q * S

class LiftFromWeight(om.ExplicitComponent):
    # for non-accelerating phases
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):        
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=nn, units='kg')
        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')
        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')
        
        self.add_input(
            name='CL', val=np.ones(nn), units='unitless'
        )

        add_aviary_output(self, Dynamic.Vehicle.LIFT, shape=nn, units='N')
        self.add_output(name='CL', shape=nn, units='unitless')

    
    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(
            Dynamic.Vehicle.LIFT, 
            Dynamic.Vehicle.MASS, 
            rows=rows_cols, 
            cols=rows_cols, 
            val=grav_metric
        )

        self.declare_partials(
            Dynamic.Vehicle.LIFT,
            [Aircraft.Wing.AREA, Dynamic.Atmosphere.DYNAMIC_PRESSURE],
            dependent=False
        )

        self.declare_partials('CL', Aircraft.Wing.AREA)
        
        self.declare_partials(
            'CL',
            [Dynamic.Vehicle.MASS, Dynamic.Atmosphere.DYNAMIC_PRESSURE],
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials(
            Dynamic.Vehicle.LIFT,
            Aircraft.Wing.AREA,
            rows=rows_cols,
            cols=rows_cols,
        )

    def compute(self, inputs, outputs):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        weight = grav_metric * inputs[Dynamic.Vehicle.MASS]

        outputs['CL'] = weight / (q * S)

    def compute_partials(self, inputs, partials):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        weight = grav_metric * inputs[Dynamic.Vehicle.MASS]

        partials['CL', Aircraft.Wing.AREA] = -weight / (q * S**2)
        partials['CL', Dynamic.Vehicle.MASS] = grav_metric / (q * S)
        partials['CL', Dynamic.Atmosphere.DYNAMIC_PRESSURE] = -weight / (q**2 * S)


class ReynoldsNumber(om.ExplicitComponent):
    # using wing characteristic length. not sure if i should be doing that
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Atmosphere.TEMPERATURE, shape=nn, units='K')
        add_aviary_input(self, Dynamic.Atmosphere.DENSITY, shape=nn, units='kg/m**3')
        add_aviary_input(self, Dynamic.Mission.VELOCITY, shape=nn, units='m/s')
        add_aviary_input(self, Aircraft.Wing.CHARACTERISTIC_LENGTH, units='m')

        self.add_output(name='dynamic_viscosity', val=np.ones(nn), units='kg/m/s')
        add_aviary_output(self, Dynamic.Atmosphere.KINEMATIC_VISCOSITY, shape=nn, units='m**2/s')
        self.add_output(name='Re_number', val=np.ones(nn), units='unitless')

    def compute(self, inputs, outputs):
        T = inputs[Dynamic.Atmosphere.TEMPERATURE]
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        V = inputs[Dynamic.Mission.VELOCITY]
        L = inputs[Aircraft.Wing.CHARACTERISTIC_LENGTH]

        mu = T**0.76  # approximation from Hoerner. probably a better way somewhere
        nu = mu / rho
        Re = (V * L) / nu

        outputs['dynamic_viscosity'] = mu
        outputs[Dynamic.Atmosphere.KINEMATIC_VISCOSITY] = nu
        outputs['Re_number'] = Re


class SkinFrictionDrag(om.ExplicitComponent):
    # for wing
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('Reynolds_number', shape=nn, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')
        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')

        self.add_output(name='Dcf', val=np.ones(nn), desc='skin friction drag', units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)
        
        self.declare_partials('Dcf', Aircraft.Wing.AREA)

        self.declare_partials(
            'Dcf',
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials(
            'Dcf',
            'Reynolds_number',
            rows=rows_cols,
            cols=rows_cols,
        )

    def compute(self, inputs, outputs):
        Re = inputs['Reynolds_number']
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        S = inputs[Aircraft.Wing.AREA]

        Cf = 1 / (3.46 * np.log10(Re) - 5.6)**2  # explicit fit of Schoenherr from Hoerner

        outputs['Dcf'] = q * S * Cf
    
    def compute_partials(self, inputs, partials):
        Re = inputs['Reynolds_number']
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        S = inputs[Aircraft.Wing.AREA]

        Cf = 1 / (3.46 * np.log10(Re) - 5.6)**2  # explicit fit of Schoenherr from Hoerner

        partials['Dcf', 'Reynolds_number'] = (-3.00531 * q * S) / (Re * (3.46 * np.log10(Re) - 5.6)**3)
        partials['Dcf', Dynamic.Atmosphere.DYNAMIC_PRESSURE] = S * Cf
        partials['Dcf', Aircraft.Wing.AREA] = q * Cf

class FuselageDrag(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # assuming fuselage Reynolds number of about 3mil at a very low Mach number
        self.add_input('fuselage_interference_factor', shape=nn, units='unitless')

        