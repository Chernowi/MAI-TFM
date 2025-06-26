import numpy as np
import random
import json
import sys

# --- Default values (from original simulation.py, kept for reference) ---
DEFAULT_SIMULATION_TIME_HOURS = 24
DEFAULT_TIME_STEP_HOURS = 0.5
DEFAULT_DOMAIN_SIZE_METERS = [10000, 10000]
DEFAULT_INITIAL_N_PARTICLES = 0
DEFAULT_SPILL_LOCATION_XY_METERS = [5000, 5000]
DEFAULT_INITIAL_SPILL_RADIUS_METERS = 100
DEFAULT_SOURCE_POINT_RADIUS_METERS = 50
DEFAULT_CONTINUOUS_SOURCE_ENABLED = False
DEFAULT_SOURCE_RATE_PER_HOUR = 0
DEFAULT_SOURCE_DURATION_HOURS = 0
DEFAULT_CURRENT_VELOCITY_XY_M_PER_HR = [0.0, 0.0]
DEFAULT_WIND_SPEED_M_PER_HR = 0.0
DEFAULT_WIND_DIRECTION_DEGREES_FROM_NORTH = 0
DEFAULT_WIND_EFFECT_FACTOR = 0.03
DEFAULT_DIFFUSION_STRENGTH_M_PER_SQRT_HR = 10.0
DEFAULT_MEAN_PARTICLE_LIFETIME_HOURS = 48
DEFAULT_PARTICLE_LIFETIME_STDDEV_HOURS = 10

# --- Helper Functions ---
def calculate_wind_induced_current(speed, direction_degrees, effect_factor):
    if speed == 0:
        return np.array([0.0, 0.0])
    math_angle_rad = np.deg2rad(90 - direction_degrees)
    return np.array([
        speed * effect_factor * np.cos(math_angle_rad),
        speed * effect_factor * np.sin(math_angle_rad)
    ])

# --- Particle Class ---
class OilParticle:
    def __init__(self, x, y, lifetime):
        self.pos = np.array([x, y], dtype=float)
        self.active = True
        self.lifetime = lifetime
        self.age = 0.0

    def update(self, dt, effective_current_vel, current_diffusion_strength, domain_size_dims):
        if not self.active:
            return
        self.pos += effective_current_vel * dt
        random_step = np.random.normal(0, current_diffusion_strength * np.sqrt(dt), 2)
        self.pos += random_step
        self.age += dt
        if self.age >= self.lifetime:
            self.active = False
        if not (0 <= self.pos[0] < domain_size_dims[0] and 0 <= self.pos[1] < domain_size_dims[1]):
            self.active = False

class OilSpillSimulatorCore:
    def __init__(self, sim_config_filepath):
        self.config_filepath = sim_config_filepath
        self.config_data = self._load_configuration(sim_config_filepath)
        self._initialize_simulation_from_config(self.config_data)

    def _load_configuration(self, filepath):
        try:
            with open(filepath, 'r') as f:
                conf_data = json.load(f)
            # print(f"Successfully loaded simulation core configuration from {filepath}")
            return conf_data
        except FileNotFoundError:
            print(f"ERROR: Simulation core configuration file not found at {filepath}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {filepath}. Check for syntax errors.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading sim core config: {e}")
            sys.exit(1)

    def _initialize_simulation_from_config(self, cfg):
        self.particles = []
        self.total_particles_generated = 0
        
        simulation_setup = cfg.get("simulation_setup", {})
        self.SIMULATION_TIME_HOURS = simulation_setup.get("simulation_time_hours", DEFAULT_SIMULATION_TIME_HOURS)
        self.TIME_STEP_HOURS = simulation_setup.get("time_step_hours", DEFAULT_TIME_STEP_HOURS)
        self.domain_dimensions_meters = np.array(simulation_setup.get("domain_size_meters", DEFAULT_DOMAIN_SIZE_METERS))

        particle_props_cfg = cfg.get("particle_properties", {})
        self.mean_particle_lifetime_hours = particle_props_cfg.get("mean_lifetime_hours", DEFAULT_MEAN_PARTICLE_LIFETIME_HOURS)
        self.particle_lifetime_stddev_hours = particle_props_cfg.get("lifetime_stddev_hours", DEFAULT_PARTICLE_LIFETIME_STDDEV_HOURS)
        self.current_diffusion_strength_m_per_sqrt_hr = particle_props_cfg.get("diffusion_strength_m_per_sqrt_hr", DEFAULT_DIFFUSION_STRENGTH_M_PER_SQRT_HR)

        initial_spill_cfg = cfg.get("initial_spill", {})
        initial_n_particles = initial_spill_cfg.get("num_particles", DEFAULT_INITIAL_N_PARTICLES)
        spill_location_xy_meters = np.array(initial_spill_cfg.get("location_xy_meters", DEFAULT_SPILL_LOCATION_XY_METERS))
        initial_spill_radius_meters = initial_spill_cfg.get("radius_meters", DEFAULT_INITIAL_SPILL_RADIUS_METERS)

        for _ in range(initial_n_particles):
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0, initial_spill_radius_meters)
            x = spill_location_xy_meters[0] + radius * np.cos(angle)
            y = spill_location_xy_meters[1] + radius * np.sin(angle)
            lifetime = random.gauss(self.mean_particle_lifetime_hours, self.particle_lifetime_stddev_hours)
            self.particles.append(OilParticle(x, y, max(0, lifetime)))
            self.total_particles_generated +=1
        
        initial_env = cfg.get("initial_environment", {})
        self.initial_current_velocity_xy_m_per_hr = np.array(initial_env.get("current_velocity_xy_m_per_hr", DEFAULT_CURRENT_VELOCITY_XY_M_PER_HR))
        self.actual_current_velocity_xy_m_per_hr = np.copy(self.initial_current_velocity_xy_m_per_hr)

        self.current_wind_speed_m_per_hr = initial_env.get("wind_speed_m_per_hr", DEFAULT_WIND_SPEED_M_PER_HR)
        self.current_wind_direction_degrees_from_north = initial_env.get("wind_direction_degrees_from_north", DEFAULT_WIND_DIRECTION_DEGREES_FROM_NORTH)
        self.wind_effect_factor = initial_env.get("wind_effect_factor", DEFAULT_WIND_EFFECT_FACTOR)
        self.current_wind_induced_current_m_per_hr = calculate_wind_induced_current(
            self.current_wind_speed_m_per_hr, self.current_wind_direction_degrees_from_north, self.wind_effect_factor
        )
        
        env_changes_all = cfg.get("environmental_changes", [])
        env_changes_all.sort(key=lambda x: x.get('time_hours', float('inf')))
        self.environmental_changes_sorted = env_changes_all
        self.next_discrete_env_change_idx = 0

        self.g_current_change_schedule = [{'time_hours': 0.0, 'velocity': np.copy(self.initial_current_velocity_xy_m_per_hr)}]
        for event in self.environmental_changes_sorted:
            if 'current_velocity_xy_m_per_hr' in event:
                self.g_current_change_schedule.append({
                    'time_hours': event['time_hours'],
                    'velocity': np.array(event['current_velocity_xy_m_per_hr'])
                })
        self.g_current_change_schedule.sort(key=lambda x: x['time_hours'])
        
        unique_time_schedule = []
        last_time = -1
        for event in reversed(self.g_current_change_schedule):
            if event['time_hours'] != last_time:
                unique_time_schedule.append(event)
                last_time = event['time_hours']
        self.g_current_change_schedule = list(reversed(unique_time_schedule))

        if not self.g_current_change_schedule or self.g_current_change_schedule[-1]['time_hours'] < self.SIMULATION_TIME_HOURS:
            last_vel = self.g_current_change_schedule[-1]['velocity'] if self.g_current_change_schedule else np.copy(self.initial_current_velocity_xy_m_per_hr)
            if not self.g_current_change_schedule or self.g_current_change_schedule[-1]['time_hours'] != self.SIMULATION_TIME_HOURS:
                 self.g_current_change_schedule.append({'time_hours': self.SIMULATION_TIME_HOURS, 'velocity': np.copy(last_vel)})

        self.g_interp_current_start_time_hr = self.g_current_change_schedule[0]['time_hours']
        self.g_interp_current_start_velocity_xy_m_per_hr = self.g_current_change_schedule[0]['velocity']
        if len(self.g_current_change_schedule) > 1:
            self.g_interp_current_end_time_hr = self.g_current_change_schedule[1]['time_hours']
            self.g_interp_current_end_velocity_xy_m_per_hr = self.g_current_change_schedule[1]['velocity']
            self.g_current_schedule_idx = 1
        else:
            self.g_interp_current_end_time_hr = self.SIMULATION_TIME_HOURS
            self.g_interp_current_end_velocity_xy_m_per_hr = np.copy(self.g_interp_current_start_velocity_xy_m_per_hr)
            self.g_current_schedule_idx = 0 
        
        self.current_simulation_time_hours = 0.0
        self.simulation_step_count = 0

    def reset(self):
        """Re-initializes the simulation to its starting state based on the loaded config."""
        self._initialize_simulation_from_config(self.config_data)

    def _add_particles_from_source(self, num_new_particles, source_cfg):
        source_location = np.array(source_cfg.get("location_xy_meters", DEFAULT_SPILL_LOCATION_XY_METERS))
        source_radius = source_cfg.get("source_point_radius_meters", DEFAULT_SOURCE_POINT_RADIUS_METERS)
        for _ in range(num_new_particles):
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0, source_radius) 
            x = source_location[0] + radius * np.cos(angle)
            y = source_location[1] + radius * np.sin(angle)
            lifetime = random.gauss(self.mean_particle_lifetime_hours, self.particle_lifetime_stddev_hours)
            self.particles.append(OilParticle(x, y, max(0, lifetime)))
            self.total_particles_generated +=1

    def step(self):
        """Advances the simulation by one internal time step."""
        if self.is_finished():
            # print("Warning: step() called on a finished simulation.")
            return

        # --- Discrete Environmental Changes (Wind, Diffusion Strength) ---
        while self.next_discrete_env_change_idx < len(self.environmental_changes_sorted) and \
              self.current_simulation_time_hours >= self.environmental_changes_sorted[self.next_discrete_env_change_idx].get('time_hours', float('inf')):
            change = self.environmental_changes_sorted[self.next_discrete_env_change_idx]
            
            wind_changed = False
            if 'wind_speed_m_per_hr' in change:
                self.current_wind_speed_m_per_hr = change['wind_speed_m_per_hr']
                wind_changed = True
            if 'wind_direction_degrees_from_north' in change:
                self.current_wind_direction_degrees_from_north = change['wind_direction_degrees_from_north']
                wind_changed = True
            if wind_changed:
                self.current_wind_induced_current_m_per_hr = calculate_wind_induced_current(
                    self.current_wind_speed_m_per_hr, self.current_wind_direction_degrees_from_north, self.wind_effect_factor
                )
            
            if 'diffusion_strength_m_per_sqrt_hr' in change:
                self.current_diffusion_strength_m_per_sqrt_hr = change['diffusion_strength_m_per_sqrt_hr']
            
            self.next_discrete_env_change_idx += 1

        # --- Interpolated Current Velocity Change ---
        while self.g_current_schedule_idx < len(self.g_current_change_schedule) and \
              self.current_simulation_time_hours >= self.g_current_change_schedule[self.g_current_schedule_idx]['time_hours']:
            
            self.g_interp_current_start_time_hr = self.g_current_change_schedule[self.g_current_schedule_idx]['time_hours']
            self.g_interp_current_start_velocity_xy_m_per_hr = np.copy(self.g_current_change_schedule[self.g_current_schedule_idx]['velocity'])
            
            self.g_current_schedule_idx += 1
            
            if self.g_current_schedule_idx < len(self.g_current_change_schedule):
                self.g_interp_current_end_time_hr = self.g_current_change_schedule[self.g_current_schedule_idx]['time_hours']
                self.g_interp_current_end_velocity_xy_m_per_hr = np.copy(self.g_current_change_schedule[self.g_current_schedule_idx]['velocity'])
            else:
                self.g_interp_current_end_time_hr = self.SIMULATION_TIME_HOURS + self.TIME_STEP_HOURS 
                self.g_interp_current_end_velocity_xy_m_per_hr = np.copy(self.g_interp_current_start_velocity_xy_m_per_hr)
                
        segment_duration = self.g_interp_current_end_time_hr - self.g_interp_current_start_time_hr
        if segment_duration <= 1e-9: 
            fraction = 1.0
        else:
            fraction = (self.current_simulation_time_hours - self.g_interp_current_start_time_hr) / segment_duration
        fraction = min(1.0, max(0.0, fraction)) 

        self.actual_current_velocity_xy_m_per_hr = self.g_interp_current_start_velocity_xy_m_per_hr + \
                                   fraction * (self.g_interp_current_end_velocity_xy_m_per_hr - self.g_interp_current_start_velocity_xy_m_per_hr)

        # --- Continuous Source ---
        source_cfg = self.config_data.get("continuous_source", {})
        if source_cfg.get("enabled", DEFAULT_CONTINUOUS_SOURCE_ENABLED) and \
           self.current_simulation_time_hours < source_cfg.get("duration_hours", DEFAULT_SOURCE_DURATION_HOURS):
            rate_per_hour = source_cfg.get("rate_per_hour", DEFAULT_SOURCE_RATE_PER_HOUR)
            particles_to_add_float = rate_per_hour * self.TIME_STEP_HOURS
            num_new_particles = int(particles_to_add_float)
            if random.random() < (particles_to_add_float - num_new_particles):
                num_new_particles += 1
            if num_new_particles > 0:
                 self._add_particles_from_source(num_new_particles, source_cfg)

        # --- Update Particles ---
        effective_advection_velocity_m_per_hr = self.actual_current_velocity_xy_m_per_hr + self.current_wind_induced_current_m_per_hr
        active_particles_list = []
        for p in self.particles:
            if p.active:
                p.update(self.TIME_STEP_HOURS, effective_advection_velocity_m_per_hr, self.current_diffusion_strength_m_per_sqrt_hr, self.domain_dimensions_meters)
                if p.active:
                    active_particles_list.append(p)
        self.particles = active_particles_list

        self.current_simulation_time_hours += self.TIME_STEP_HOURS
        self.simulation_step_count += 1
    
    def get_active_particle_positions(self):
        """Returns a list of [x,y] meter coordinates for active particles."""
        return [p.pos for p in self.particles if p.active]

    def get_current_environmental_current(self):
        """Returns the (vx, vy) current vector in meters/hour for the current simulation time."""
        # This is the base ocean current, not including wind-induced effects
        return np.copy(self.actual_current_velocity_xy_m_per_hr)

    def get_current_wind_induced_current(self):
        """Returns the (vx, vy) wind-induced current vector in meters/hour."""
        return np.copy(self.current_wind_induced_current_m_per_hr)
        
    def get_total_effective_current(self):
        """Returns the combined (vx,vy) ocean + wind-induced current in m/hr."""
        return self.actual_current_velocity_xy_m_per_hr + self.current_wind_induced_current_m_per_hr

    def is_finished(self):
        """Returns true if simulation duration is reached."""
        return self.current_simulation_time_hours >= self.SIMULATION_TIME_HOURS

    def get_current_sim_time_hours(self):
        return self.current_simulation_time_hours

    def get_time_step_hours(self):
        return self.TIME_STEP_HOURS

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy config file for testing
    dummy_config = {
        "simulation_setup": {
            "simulation_time_hours": 2, # Short sim for test
            "time_step_hours": 0.1,
            "domain_size_meters": [1000, 1000],
        },
        "initial_spill": {
            "num_particles": 10,
            "location_xy_meters": [500, 500],
            "radius_meters": 50
        },
        "particle_properties": {
            "mean_lifetime_hours": 5,
            "lifetime_stddev_hours": 1,
            "diffusion_strength_m_per_sqrt_hr": 5.0
        },
        "initial_environment": {
            "current_velocity_xy_m_per_hr": [10.0, 5.0], # m/hr
            "wind_speed_m_per_hr": 20.0, # m/hr
            "wind_direction_degrees_from_north": 45,
            "wind_effect_factor": 0.03
        },
        "environmental_changes": [
            {
                "time_hours": 0.5,
                "current_velocity_xy_m_per_hr": [0.0, -10.0]
            },
            {
                "time_hours": 1.0,
                "wind_speed_m_per_hr": 0.0
            }
        ]
    }
    config_path = "dummy_sim_core_config.json"
    with open(config_path, 'w') as f:
        json.dump(dummy_config, f, indent=4)

    print(f"Created dummy config: {config_path}")
    simulator = OilSpillSimulatorCore(config_path)
    print(f"Simulator initialized. Domain: {simulator.domain_dimensions_meters}m. Sim Time: {simulator.SIMULATION_TIME_HOURS}h. Time step: {simulator.TIME_STEP_HOURS}h")

    step = 0
    while not simulator.is_finished():
        simulator.step()
        active_particles = simulator.get_active_particle_positions()
        current_vel = simulator.get_current_environmental_current()
        wind_current = simulator.get_current_wind_induced_current()
        total_current = simulator.get_total_effective_current()
        sim_time = simulator.get_current_sim_time_hours()
        print(f"Step {step}: Time {sim_time:.2f}h, Active Particles: {len(active_particles)}, Ocean Current: {current_vel}, Wind Current: {wind_current}, TotalEff: {total_current}")
        if len(active_particles) > 0:
            print(f"  First particle pos: {active_particles[0]}")
        step += 1
    
    print("Simulation core test finished.")