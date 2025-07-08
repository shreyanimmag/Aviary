import numpy as np

import openmdao.api as om

from ambiance import Atmosphere

from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag
from openaerostruct.meshing.mesh_generator import generate_mesh

from aviary.api import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

class ReynoldsNumber(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Mission.ALTITUDE, shape=nn, units='m')
        add_aviary_input(self, Dynamic.Mission.VELOCITY, shape=nn, units='m/s')
        add_aviary_input(self, Aircraft.Wing.CHARACTERISTIC_LENGTH, units='m')

        add_aviary_output(self, Dynamic.Atmosphere.TEMPERATURE, shape=nn, units='K')
        add_aviary_output(self, Dynamic.Atmosphere.KINEMATIC_VISCOSITY, shape=nn, units='m**2/s')
        add_aviary_output(self, Dynamic.Atmosphere.DENSITY, shape=nn, units='kg/m**3')

        self.add_output(name='dynamic_viscosity', shape=nn, units='kg/m/s')
        self.add_output(name='Reynolds_number', shape=nn, units='unitless')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        V = inputs[Dynamic.Mission.VELOCITY]
        h = inputs[Dynamic.Mission.ALTITUDE]
        L = inputs[Aircraft.Wing.CHARACTERISTIC_LENGTH]
        
        T = np.zeros(nn)
        rho = np.zeros(nn)
        mu = np.zeros(nn)

        for i in range(nn):
            atmosphere = Atmosphere(h[i])
            T[i] = atmosphere.temperature
            rho[i] = atmosphere.density
            mu[i] = atmosphere.dynamic_viscosity
        
        nu = mu / rho

        outputs[Dynamic.Atmosphere.TEMPERATURE] = T
        outputs[Dynamic.Atmosphere.DENSITY] = rho
        outputs['dynamic_viscosity'] = mu
        outputs[Dynamic.Atmosphere.KINEMATIC_VISCOSITY] = nu

        Re = V * L / nu

        outputs['Reynolds_number'] = Re

class OASAero(om.Group):

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("aviary_inputs")

    def setup(self):
        num_nodes = self.options["num_nodes"]
        aviary_inputs = self.options["aviary_inputs"]

        altitude = aviary_inputs.get_val(Dynamic.Mission.ALTITUDE, units="ft")
        velocity = aviary_inputs.get_val(Dynamic.Mission.VELOCITY, units="ft/s")
        root_chord = aviary_inputs.get_val(Aircraft.Wing.ROOT_CHORD, units="ft")
        span = aviary_inputs.get_val(Aircraft.Wing.SPAN, units="ft")

        atm = Atmosphere(altitude)
        rho = atm.density  # slug/ft³
        mu = atm.dynamic_viscosity  # slug/(ft·s)

        self.add_subsystem(
            "reynolds_calc",
            ReynoldsNumber(num_nodes=num_nodes),
            promotes_outputs=["Re"]
        )

        self.set_input_defaults("altitude", altitude, units="ft")
        self.set_input_defaults("velocity", velocity, units="ft/s")
        self.set_input_defaults("density", rho, units="slug/ft**3")
        self.set_input_defaults("dynamic_viscosity", mu, units="slug/(ft*s)")
        self.set_input_defaults("char_length", root_chord, units="ft")

        mesh_dict = {
            "num_y": 5,
            "num_x": 2,
            "wing_type": "rect",
            "symmetry": True,
            "span": span,
            "chord": root_chord,
        }
        mesh = generate_mesh(mesh_dict)

        surface = {
            "name": "wing",
            "symmetry": True,
            "S_ref_type": "wetted",
            "mesh": mesh,
            "t_over_c_cp": np.array([
                aviary_inputs.get_val(Aircraft.Wing.THICKNESS_TO_CHORD)
            ]),
            "c_max_t": aviary_inputs.get_val(Aircraft.Wing.MAX_THICKNESS_LOCATION),
        }

        aero_point = AeroPoint(
            surfaces=[surface],
            compute_partials=True,
        )

        self.add_subsystem(
            "aero_point_0",
            aero_point,
            promotes_inputs=["alpha", "v", "rho"],
            promotes_outputs=["CL", "CD"],
        )

        self.connect("Re", "aero_point_0.wing.Re")

        self.set_input_defaults("alpha", val=5.0, units="deg")
        self.set_input_defaults("v", val=velocity, units="ft/s")
        self.set_input_defaults("rho", val=rho, units="slug/ft**3")