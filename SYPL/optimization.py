"""
Python code for the floating wind project development based on:

Van Heukelum, H. J., Binnekamp, R., & Wolfert, A. R. M. (2024). Socio-technical systems in-
tegration and design: a multi-objective optimisation method based on integrative preference
maximisation. Structure and Infrastructure Engineering (https://doi.org/10.1080/15732479.2023.2297891). 
"""

from math import ceil
import numpy as np
from numpy import pi
from scipy.interpolate import pchip_interpolate
from scipy.optimize import fsolve
from random import seed

from preferendus import GeneticAlgorithm

"""
Before starting the SODO and MODO some general variables and characteristics are set.
Reference for the ship characteristics:
https://www.oilandgasiq.com/drilling-and-development/articles/offshore-support-vessels-leading-emissions-reducti
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import pchip_interpolate
from math import pi, ceil


from scipy.stats import beta

class OptimizationClass:
    def __init__(self,
                 iteration_seed,
                 w1,
                 w2,
                 w3,
                 w4,
                 x_points_1,
                 p_points_1,
                 x_points_2,
                 p_points_2,
                 x_points_3,
                 p_points_3,
                 x_points_4,
                 p_points_4,
    ):
        self.iteration_seed = iteration_seed
        
        # Set weights for the different objectives
        self.w1 = w1  # project duration
        self.w2 = w2  # cost
        self.w3 = w3  # fleet utilization
        self.w4 = w4  # sustainability

        # Set preference points
        self.x_points_1 = x_points_1
        self.p_points_1 = p_points_1
        self.x_points_2 = x_points_2
        self.p_points_2 = p_points_2
        self.x_points_3 = x_points_3
        self.p_points_3 = p_points_3
        self.x_points_4 = x_points_4
        self.p_points_4 = p_points_4

        self.max_t = 3800
        self.n_anchors = 108
        self.constants = {
            'NC': 9,
            'NQ': 1,
            'W_steel': 78.5,  # kN/m3
            'W_water': 10.25,  # kN/m3
            'W_concrete': 25,  # kN/m3
        }
        self.ship_options = {
            'OCV small': {
                'day_rate': 47000,  # €/day
                'deck_space': 8,  # max anchors on deck
                'max_available': 3, # how many of this type are available for the project
                'CO2_emission': 30,  # tonnes per day
                'chance': 0.7
            },
            'OCV big': {
                'day_rate': 55000,
                'deck_space': 12,
                'max_available': 2,
                'CO2_emission': 40,  # tonnes per day
                'chance': 0.8
            },
            'Barge': {
                'day_rate': 35000,
                'deck_space': 16,
                'max_available': 2,
                'CO2_emission': 35,  # tonnes per day
                'chance': 0.5
            }
        }
        self.soil_data = {
            'type': 'clay',
            'su': 60,  # kPa (Undrained shear strength)
            'a_i': 0.64, # - (Coefficient of shaft friction)
            'a_o': 0.64,
            'sat_weight': 9,  # kN/m3 (Submerged unit weight)
        }
        self.mooring_data = {
            'type': 'catenary',
            'line type': 'chain',
            'd': 0.24,  # m (Nominal chain diameter)
            'mu': 0.25,  # - (Coefficient of seabed friction)
            'AWB': 2.5  # - (Active bearing area coefficient)
        }

        self.time_installation = 1  # time it takes to install one anchor
        self.time_bunkering = [1.5, 2, 2.5]  # time it takes to get a new set of anchors onboard
        self.risk_events = {
            'weather_delay': {
                'probability': 0.2,
                'impact': lambda: (self.generate_beta_pert_duration(0.5, 1, 1.5), 'installation'),
            },
            'supply_chain_issue': {
                'probability': 0.1,
                'impact': lambda: (self.generate_beta_pert_duration(1, 1.5, 2), 'bunkering'),
            },
            'technical_failure': {
                'probability': 0.15,
                'impact': lambda: (self.generate_beta_pert_duration(0.5, 1, 1.5), 'installation'),
            },
            'logistical_constraints': {
                'probability': 0.2,
                'impact': lambda: (self.generate_beta_pert_duration(1, 1.5, 3), 'bunkering'),
            },
            'marine_traffic': {
                'probability': 0.05,
                'impact': lambda: (self.generate_beta_pert_duration(0.5, 0.75, 2), 'installation'),
            },
            'environmental_restrictions': {
                'probability': 0.1,
                'impact': lambda: (self.generate_beta_pert_duration(2, 3.5, 5), 'bunkering'),
            },
            'lack_of_skilled_personnel': {
                'probability': 0.08,
                'impact': lambda: (self.generate_beta_pert_duration(1, 2, 3), 'installation'),
            },
            'equipment_shortage': {
                'probability': 0.12,
                'impact': lambda: (self.generate_beta_pert_duration(1, 2, 4), 'installation'),
            }
        }

        # Bounds and constraints
        self.bounds = [
            [0, self.ship_options['OCV small']['max_available']],
            [0, self.ship_options['OCV big']['max_available']],
            [0, self.ship_options['Barge']['max_available']],
            [1.5, 4],
            [2, 8]
        ]
        self.cons = [('ineq', self.constraint_1), ('ineq', self.constraint_2)]

        self.simulation_cache = {}

    def generate_beta_pert_duration(self, min_duration, most_likely_duration, max_duration):
        """
        Generates a random duration based on the beta-PERT distribution.
        
        :param min_duration: Minimum possible duration
        :param most_likely_duration: Most likely duration
        :param max_duration: Maximum possible duration
        :return: A random duration value from the beta-PERT distribution
        """
        # Calculate the shape parameters for the beta distribution
        a = 1 + 4 * ((most_likely_duration - min_duration) / (max_duration - min_duration))
        b = 1 + 4 * ((max_duration - most_likely_duration) / (max_duration - min_duration))
        
        # Scale and location parameters
        scale = max_duration - min_duration
        loc = min_duration
        
        # Generate a random value from the beta distribution and scale it
        duration = beta(a, b).rvs() * scale + loc
        return duration

    def objective_time(self, ocv_s, ocv_l, barge):
        """
        Function to calculate the project duration

        :param ocv_s: number of small offshore construction vessels
        :param ocv_l: number of large offshore construction vessels
        :param barge: number of barges
        :return: overall project duration, ocv_s time, ocv_l time, barge time
        """
        # set empty list for respective vessel time
        t_array = list()
        t_ocv_s = list()
        t_ocv_l = list()
        t_barge = list()

        # loop through all diffrent combinations of ships
        for ip in range(len(ocv_s)):

            # Create a unique key for the current configuration
            config_key = (ocv_s[ip], ocv_l[ip], barge[ip])

            # Reset the seed for each simulation run within this optimization to ensure the same
            # risk probability, risk duration, and activty duration for every simulation within the optimization
            np.random.seed(self.iteration_seed)
            seed(self.iteration_seed)

            if config_key in self.simulation_cache:
                # If the result is cached, use it
                time_ocv_s, time_ocv_l, time_barge = self.simulation_cache[config_key]

            else:

                # initialize timers and counter
                inf_loop_prevent = 0
                time_ocv_s = 0
                time_ocv_l = 0
                time_barge = 0
                anchor_counter = 0

                # define ship deck space option
                ds_ocv_s = self.ship_options['OCV small']['deck_space']
                ds_ocv_l = self.ship_options['OCV big']['deck_space']
                ds_barge = self.ship_options['Barge']['deck_space']

                # iteration through installation process as long as number of anchor to install is smaller than
                # the number of anchors available
                while self.n_anchors - anchor_counter > 0:
                    # Check for and apply installation-related risk event delays
                    installation_delay_ocv_s, active_risks_ocv_s = self.check_risk_events('installation')
                    installation_delay_ocv_l, active_risks_ocv_l = self.check_risk_events('installation')
                    installation_delay_barge, active_risks_barge = self.check_risk_events('installation')

                    time_ocv_s += installation_delay_ocv_s
                    time_ocv_l += installation_delay_ocv_l
                    time_barge += installation_delay_barge

                    # check if ships are fully or only partially loaded
                    if self.n_anchors - anchor_counter < ocv_s[ip] * ds_ocv_s + ocv_l[ip] * ds_ocv_l + barge[ip] * ds_barge:
                        n = ocv_s[ip] + ocv_l[ip] + barge[ip]  # Number of vessels
                        # number of anchors left is number of remaining anchors divided by number of ships (oversimplification)
                        anchors_left_per_vessel = ceil((self.n_anchors - anchor_counter) / n)
                        diff_1 = 0
                        diff_2 = 0
                        ds_ocv_s = anchors_left_per_vessel
                        ds_ocv_l = anchors_left_per_vessel
                        ds_barge = anchors_left_per_vessel

                        # distribute remaining number of anchors on ship starting with the OCV small
                        # if number of remaining anchors to store is smaller then the deck space of the OCV small
                        # are distributed on to the OCV big
                        if ds_ocv_s > self.ship_options['OCV small']['deck_space']:
                            diff_1 = ocv_s[ip] * (anchors_left_per_vessel - self.ship_options['OCV small']['deck_space'])
                            ds_ocv_s = self.ship_options['OCV small']['deck_space']

                            # if the deck space of the OCV big is exceede the remaining anchors are stored on the barge
                            if ocv_l[ip] != 0:
                                if ds_ocv_l + diff_1 / ocv_l[ip] > self.ship_options['OCV big']['deck_space']:
                                    diff_2 = ocv_l[ip] * (
                                            anchors_left_per_vessel + round(diff_1 / ocv_l[ip]) - self.ship_options['OCV big'][
                                        'deck_space'])
                                    ds_ocv_l = self.ship_options['OCV big']['deck_space']
                                    ds_barge += diff_2 / barge[ip]
                                else:
                                    ds_ocv_l = anchors_left_per_vessel + ceil(diff_1 / (ocv_l[ip] + barge[ip]))
                                    ds_barge = anchors_left_per_vessel + ceil(diff_1 / (ocv_l[ip] + barge[ip]))
                            else:
                                ds_barge = anchors_left_per_vessel + ceil(diff_1 / barge[ip])

                        # check that the number of anchors to be installed is equal to or greater than the number of anchors
                        # remaining to be installed
                        try:
                            assert ocv_s[ip] * ds_ocv_s + ocv_l[ip] * ds_ocv_l + barge[ip] * ds_barge >= \
                                (self.n_anchors - anchor_counter)
                        
                        # if not the code will be stopped
                        except AssertionError as err:
                            print(ocv_s[ip], ocv_l[ip], barge[ip])
                            print(self.n_anchors - anchor_counter)
                            print(anchors_left_per_vessel)
                            print(diff_1)
                            print(diff_2)
                            print(ds_ocv_s)
                            print(ds_ocv_l)
                            print(ds_barge)
                            raise err

                    # Generate variable durations for installation
                    var_duration_ocv_s = self.generate_beta_pert_duration(
                        0.8 * self.time_installation, 
                        self.time_installation, 
                        1.2 * self.time_installation)
                    
                    var_duration_ocv_l = self.generate_beta_pert_duration(
                        0.7 * self.time_installation, 
                        self.time_installation, 
                        1.6 * self.time_installation)
                    
                    var_duration_barge = self.generate_beta_pert_duration(
                        0.6 * self.time_installation, 
                        self.time_installation, 
                        1.9 * self.time_installation)
                    
                    #  increase respective ship time
                    time_ocv_s += ocv_s[ip] * ds_ocv_s * var_duration_ocv_s
                    time_ocv_l += ocv_l[ip] * ds_ocv_l * var_duration_ocv_l
                    time_barge += barge[ip] * ds_barge * var_duration_barge

                    # increase anchor counter to number of installed anchors
                    anchor_counter += ocv_s[ip] * self.ship_options['OCV small']['deck_space'] + ocv_l[ip] * self.ship_options['OCV big'][
                        'deck_space'] + barge[ip] * self.ship_options['Barge']['deck_space']

                    if self.n_anchors - anchor_counter <= 0:  # check if it is still the case after installation of last anchors

                        bunkering_delay_ocv_s, active_risks_bunkering_ocv_s = self.check_risk_events('bunkering')
                        bunkering_delay_ocv_l, active_risks_bunkering_ocv_l = self.check_risk_events('bunkering')
                        bunkering_delay_barge, active_risks_bunkering_barge = self.check_risk_events('bunkering')

                        var_duration_ocv_s = self.generate_beta_pert_duration(
                            0.6 * self.time_bunkering[0], 
                            self.time_bunkering[0], 
                            1.3 * self.time_bunkering[0])
                    
                        var_duration_ocv_l = self.generate_beta_pert_duration(
                            0.6 * self.time_bunkering[1], 
                            self.time_bunkering[1], 
                            1.3 * self.time_bunkering[1])
                        
                        var_duration_barge = self.generate_beta_pert_duration(
                            0.6 * self.time_bunkering[2], 
                            self.time_bunkering[2], 
                            1.3 * self.time_bunkering[2])

                        time_ocv_s += ocv_s[ip] * var_duration_ocv_s + bunkering_delay_ocv_s # add time to load new anchor
                        time_ocv_l += ocv_l[ip] * var_duration_ocv_l + bunkering_delay_ocv_l
                        time_barge += barge[ip] * var_duration_barge + bunkering_delay_barge

                    inf_loop_prevent += 1  # preventing an infinite loop (if sum of ships is zero)

                    # if no anchors are installed, the while loop returns a high value for the timers and breaks the while loop
                    if inf_loop_prevent > 20:
                        time_ocv_s += 1e4
                        time_ocv_l += 1e4
                        time_barge += 1e4
                        break

                    # Cache the result
                    self.simulation_cache[config_key] = [time_ocv_s, time_ocv_l, time_barge]

            # time is added to overall list of alternatives
            t_ocv_s.append(time_ocv_s)
            t_ocv_l.append(time_ocv_l)
            t_barge.append(time_barge)
            t_array.append(max(time_ocv_s, time_ocv_l, time_barge))

        # # Print the cache for debugging
        # print("Cache contents:")
        # for key, value in self.simulation_cache.items():
        #     print(f"Config: {key}, Time: {value}")

        return t_array, t_ocv_s, t_ocv_l, t_barge


    def check_risk_events(self, activity_type):
        """
        Checks for risk events related to a specific activity type, calculates the additional delay,
        and stores the risks that are triggered.
        
        :param activity_type: The type of activity to check risk events for ('installation' or 'bunkering').
        :return: A tuple with the additional delay for the specified activity type and a list of triggered risk events.
        """
        additional_delay = 0
        triggered_risks = []  # List to store the names of triggered risk events

        for event, details in self.risk_events.items():
            if np.random.random() < details['probability']:  # Check if the event occurs
                delay, affected_activity = details['impact']()  # Get the delay and affected activity
                if affected_activity == activity_type:
                    additional_delay += float(delay)
                    triggered_risks.append(event)  # Store the event nam
        return additional_delay, triggered_risks



    def objective_costs(self, diameter, length, t_ocv_s, t_ocv_l, t_barge):
        """Function to calculate the installation costs"""

        t = 0.02 * diameter
        mass_steel = (pi * length * diameter * t + pi / 4 * diameter ** 2 * t) * 7.85  # mT
        production_costs_anchor = (mass_steel * 815 + 40000) * self.n_anchors  # Calculate material cost

        costs_ocv_s = np.array(t_ocv_s) * self.ship_options['OCV small']['day_rate']
        costs_ocv_l = np.array(t_ocv_l) * self.ship_options['OCV big']['day_rate']
        costs_barge = np.array(t_barge) * self.ship_options['Barge']['day_rate']
        return production_costs_anchor + costs_ocv_s + costs_ocv_l + costs_barge


    def objective_fleet_utilization(self, ocv_s, ocv_l, barge):
        """Function to calculate the fleet utilization"""
        chance_ocv_s = self.ship_options['OCV small']['chance'] ** ocv_s
        chance_ocv_l = self.ship_options['OCV big']['chance'] ** ocv_l
        chance_barge = self.ship_options['Barge']['chance'] ** barge
        return np.prod([np.power(chance_ocv_s, ocv_s), np.power(chance_ocv_l, ocv_l), np.power(chance_barge, barge)],
                    axis=0)


    def objective_co2(self, ocv_s, ocv_l, barge, t_ocv_s, t_ocv_l, t_barge):
        """Function to calculate the CO2 emissions"""
        co2_emission_ocv_s = np.array(t_ocv_s) * self.ship_options['OCV small']['CO2_emission'] * ocv_s
        co2_emission_ocv_l = np.array(t_ocv_l) * self.ship_options['OCV big']['CO2_emission'] * ocv_l
        co2_emission_barge = np.array(t_barge) * self.ship_options['Barge']['CO2_emission'] * barge
        return co2_emission_ocv_s + co2_emission_ocv_l + co2_emission_barge


    def check_p_score(self, p):
        """Function to mak sure all preference scores are in [0,100]"""
        mask1 = p < 0
        mask2 = p > 100
        p[mask1] = 0
        p[mask2] = 100
        return p


    def objective(self, variables):
        """Objective function for the GA. Calculates all sub-objectives and their corresponding preference scores. The
        aggregation is done in the GA"""
        n_ocv_s = variables[:, 0]  # number of small offshore construction vessels
        n_ocv_l = variables[:, 1]  # number of large offshore construction vessels
        n_barge = variables[:, 2]  # number of barges
        diameter = variables[:, 3]  # anchor diameter
        length = variables[:, 4]  # anchor length

        project_time, time_ocv_s, time_ocv_l, time_barge = self.objective_time(n_ocv_s, n_ocv_l, n_barge)
        costs = self.objective_costs(diameter, length, time_ocv_s, time_ocv_l, time_barge)
        fleet_util = self.objective_fleet_utilization(n_ocv_s, n_ocv_l, n_barge)
        co2_emission = self.objective_co2(n_ocv_s, n_ocv_l, n_barge, time_ocv_s, time_ocv_l, time_barge)

        # calculate the preference scores including the check of the preference score
        p_1 = self.check_p_score(pchip_interpolate(self.x_points_1, self.p_points_1, project_time))
        p_2 = self.check_p_score(pchip_interpolate(self.x_points_2, self.p_points_2, costs))
        p_3 = self.check_p_score(pchip_interpolate(self.x_points_3, self.p_points_3, fleet_util))
        p_4 = self.check_p_score(pchip_interpolate(self.x_points_4, self.p_points_4, co2_emission))

        # aggregate preference scores and return this to the GA
        return [self.w1, self.w2, self.w3, self.w4], [p_1, p_2, p_3, p_4]

    def constraint_1(self, variables):
        """Constraint that ensures there is at least one vessel on the project"""
        n_ocv_s = variables[:, 0]
        n_ocv_l = variables[:, 1]
        n_barge = variables[:, 2]

        return -1 * (n_ocv_s + n_ocv_l + n_barge) + 1  # < 0


    def _solve_ta_ta(self, p, tension_mudline, theta_m, za, d, mu, su, nc=7.6):
        """Solve the force o the anchor and its angle, based on the tension and angle of the mooring line at the seabed"""
        tension_a, theta = p
        za_q = self.mooring_data['AWB'] * d * nc * su * za

        return (2 * za_q / tension_a) - (theta ** 2 - theta_m ** 2), np.exp(
            mu * (theta - theta_m)) - tension_mudline / tension_a


    def constraint_2(self, variables):
        """Constraint that checks if the pull force on the anchors is lower than the resistance of the anchors to this
        force.

        The calculations are based on:
            - Houlsby, G. T. and Byrne, B. W. (2005). “Design procedures for installation of suction caissons in clay and
            other materials.” Proceedings of the Institution of Civil Engineers-Geotechnical Engineering, 158(2), 75–82.
            - Randolph, M. and Gourvenec, S. (2017). Offshore geotechnical engineering. CRC press.
            - Arany, L. and Bhattacharya, S. (2018). “Simplified load estimation and sizing of suction anchors for spar buoy
            type floating offshore wind turbines.” Ocean Engineering, 159, 348–357.
        """
        diameter = variables[:, 3]
        length = variables[:, 4]

        t = 0.02 * diameter
        d_i = diameter - t
        d_e = diameter + t
        mean_diameter = diameter
        weight_anchor = (np.pi * length * mean_diameter * t + np.pi * mean_diameter ** 2 * t / 4) * (
                self.constants['W_steel'] - self.constants['W_water'])

        weight_plug = np.pi / 4 * d_i ** 2 * length * self.soil_data['sat_weight']

        external_shaft_fric = np.pi * d_e * length * self.soil_data['a_o'] * self.soil_data['su']
        internal_shaft_fric = np.pi * d_i * length * self.soil_data['a_i'] * self.soil_data['su']
        reverse_end_bearing = 6.7 * self.soil_data['su'] * d_e ** 2 * np.pi / 4

        v_mode_1 = weight_anchor + external_shaft_fric + reverse_end_bearing
        v_mode_2 = weight_anchor + external_shaft_fric + internal_shaft_fric
        v_mode_3 = weight_anchor + external_shaft_fric + weight_plug

        v_max = np.amin([v_mode_1, v_mode_2, v_mode_3], axis=0)
        h_max = length * d_e * 10 * self.soil_data['su']

        rel_pos_pad_eye = 0.5

        tension_pad_eye = np.zeros(len(length))
        angle_pad_eye = np.zeros(len(length))
        for lng in np.unique(length):
            x = fsolve(self._solve_ta_ta, np.array([10000, 1]),
                    (self.max_t, 0, rel_pos_pad_eye * lng, self.mooring_data['d'], self.mooring_data['mu'], self.soil_data['su'], 12))
            mask = length == lng
            tension_pad_eye[mask] = x[0]
            angle_pad_eye[mask] = x[1]

        h = np.cos(angle_pad_eye) * tension_pad_eye
        v = np.sin(angle_pad_eye) * tension_pad_eye

        a = length / mean_diameter + 0.5
        b = length / (3 * mean_diameter) + 4.5

        hor_util = h / h_max
        ver_util = v / v_max

        return (hor_util ** a + ver_util ** b) - 1


    def print_results(self, res):
        """Function that prints the results of the optimizations"""
        print(f'Optimal result for:\n')
        print(f'\t {res[0]} small Offshore Construction Vessels\n')
        print(f'\t {res[1]} large Offshore Construction Vessels\n')
        print(f'\t {res[2]} Barges\n')
        print(f'\tAn anchor diameter of {round(res[3], 2)}m\n')
        print(f'\tAn anchor length of {round(res[4], 2)}m\n')


    # Optimization function
    def optimization_function(self):
        """
        Function that initiates the GA.
        """

        # make dictionary with parameter settings for the GA
        options = {
            'n_bits': 20,
            'n_iter': 500,
            'n_pop': 500,
            'r_cross': 0.85,
            'max_stall': 10,
            'aggregation': 'IMAP',
            'lsd_aggregation': "fast",
            'var_type_mixed': ['int', 'int', 'int', 'real', 'real'],
        }

        # list to save the results from every run to
        save_array_IMAP = list()

        # initiate and run the GA
        ga = GeneticAlgorithm(objective=self.objective, constraints=self.cons, bounds=self.bounds, options=options)
        result , design_variables_IMAP, _ = ga.run(verbose=False)

        save_array_IMAP.append(design_variables_IMAP)

        # make numpy array of results, to allow for array splicing
        variable = np.array(save_array_IMAP)

        return variable