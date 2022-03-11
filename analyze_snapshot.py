#!/usr/bin/env python

import h5py
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from orbits import angular_momentum, node_vector, eccentricity_vector, specific_orbital_energy, elements_from_state_vector

class analyze_snapshot:
    "Class for analyzing snapshot hdf5 data."
    
    def __init__(self, fname):
        self.f = h5py.File(fname, 'r')
        self.header = self.f['Header']
        
        # Do star particles exist in this snapshot?
        self.stars_exist = False
        if 'PartType5' in self.f:
            self.stars_exist = True
            
        # Header attributes.
        self.box_size = self.header.attrs['BoxSize']
        self.sf_density_threshold = self.header.attrs['Density_Threshold_For_SF_CodeUnits']
        self.eff_ker_nn = self.header.attrs['Effective_Kernel_NeighborNumber']
        
        # Number of particles, current time.
        self.num_particles = self.header.attrs['NumPart_Total']
        self.num_p0 = self.num_particles[0]
        self.num_p5 = self.num_particles[5]
        self.t = self.header.attrs['Time']
        
        # Code units.
        self.G = self.header.attrs['Gravitational_Constant_In_Code_Inits']
        self.length = self.header.attrs['UnitLength_In_CGS']
        self.mass = self.header.attrs['UnitMass_In_CGS']
        self.velocity = self.header.attrs['UnitVelocity_In_CGS']
        
        # PartType0 data.
        self.p0 = self.f['PartType0']
            
        # Particle IDs.
        self.p0_ids = self.p0['ParticleIDs'][:]
        
        # Masses.
        self.p0_mass = self.p0['Masses'][:]
        
        # Coordinates.
        self.p0_x = self.p0['Coordinates'][:, 0]
        self.p0_y = self.p0['Coordinates'][:, 1]
        self.p0_z = self.p0['Coordinates'][:, 2]
        
        # Velocities.
        self.p0_u = self.p0['Velocities'][:, 0]
        self.p0_v = self.p0['Velocities'][:, 1]
        self.p0_w = self.p0['Velocities'][:, 2]
        
        self.p0_v2    = self.p0_u**2 + self.p0_v**2 + self.p0_w**2   # Velocity squared.
        self.p0_e_kin = 0.5 * np.multiply(self.p0_mass, self.p0_v2)  # Kinetic energy.
        
        # PartType5 data.
        self.p5 = None
        if self.stars_exist:
            self.p5 = self.f['PartType5']
            
            # Particle IDs.
            self.p5_ids = self.p5['ParticleIDs'][:]
            
            # Masses.
            self.p5_mass = self.p5['Masses'][:]
            self.p5_mu   = self.G * self.p5_mass  # Standard gravitational parameter.
        
            # Coordinates.
            self.p5_x = self.p5['Coordinates'][:, 0]
            self.p5_y = self.p5['Coordinates'][:, 1]
            self.p5_z = self.p5['Coordinates'][:, 2]
        
            # Velocities.
            self.p5_u = self.p5['Velocities'][:, 0]
            self.p5_v = self.p5['Velocities'][:, 1]
            self.p5_w = self.p5['Velocities'][:, 2]
            
            self.p5_v2    = self.p5_u**2 + self.p5_v**2 + self.p5_w**2   # Velocity squared.
            self.p5_e_kin = 0.5 * np.multiply(self.p5_mass, self.p5_v2)  # Kinetic energy.
            
    # -------------- FUNCTIONS ---------------
    
    # Calculate relative positions of gas particle with respect to the point
    # specified by (x0, y0, z0).
    def get_relative_position_from_point(self, x0, y0, z0, gas_id):
        
        # Index corresponding to chosen gas id.
        idx_g = np.where(self.p0_ids == gas_id)[0][0]
        
        # Relative position.
        x_rel = self.p0_x[idx_g] - x0
        y_rel = self.p0_y[idx_g] - y0
        z_rel = self.p0_z[idx_g] - z0
        
        return x_rel, y_rel, z_rel
    
    # Calculate relative velocities of gas particle with respect to the velocity
    # specified by (u0, v0, w0).
    def get_relative_velocity_from_point(self, u0, v0, w0, gas_id):
        
        # Index corresponding to chosen gas id.
        idx_g = np.where(self.p0_ids == gas_id)[0][0]
        
        # Relative velocity.
        u_rel = self.p0_u[idx_g] - u0
        v_rel = self.p0_v[idx_g] - v0
        w_rel = self.p0_w[idx_g] - w0
        
        return u_rel, v_rel, w_rel
    
    # Calculate relative separation between gas particle and sink particle 
    # given the sink particle ID.
    def get_distance_from_sink(self, sink_id, gas_id):
        
        # Only calculate if star particle exists.
        if self.num_p5 == 0:
            return
        
        # Index corresponding to chosen sink/gas id.
        idx_s = np.where(self.p5_ids == sink_id)[0][0]
        idx_g = np.where(self.p0_ids == gas_id)[0][0]
        
        # Relative position.
        x_rel = self.p0_x[idx_g] - self.p5_x[idx_s]
        y_rel = self.p0_y[idx_g] - self.p5_y[idx_s]
        z_rel = self.p0_z[idx_g] - self.p5_z[idx_s]
        r     = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        return r
    
    # Calculate relative distance of a gas particle from the point
    # specified by (x0, y0, z0).
    def get_distance_from_point(self, x0, y0, z0, gas_id):
        
        # Index corresponding to chosen gas id.
        idx_g = np.where(self.p0_ids == gas_id)[0][0]
        
        # Relative position.
        x_rel = self.p0_x[idx_g] - x0
        y_rel = self.p0_y[idx_g] - y0
        z_rel = self.p0_z[idx_g] - z0
        r     = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        return r
    
    # Calculate relative distance from point (x0, y0, z0) for all
    # gas particles specified by gas_ids.
    def get_distances_from_point(self, x0, y0, z0, gas_ids):
        
        # Indices corresponding to chosen gas ids.
        idx_g = np.isin(self.p0_ids, gas_ids)
        
        # Relative position.
        x_rel = self.p0_x[idx_g] - x0
        y_rel = self.p0_y[idx_g] - y0
        z_rel = self.p0_z[idx_g] - z0
        r     = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        return r
        
    
    # Determine whether gas particles are individually gravitationally bound to a
    # system with mass mc and coordinates (xc, yc, zc), (uc, vc, wc).
    def is_gravitationally_bound_to_system(self, mc, xc, yc, zc, uc, vc, wc, gas_id):
        
        # Index corresponding to chosen gas id.
        idx_g = np.where(self.p0_ids == gas_id)[0][0]
        
        # Relative position.
        x_rel = self.p0_x[idx_g] - xc
        y_rel = self.p0_y[idx_g] - yc
        z_rel = self.p0_z[idx_g] - zc
        r     = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        # Relative velocity.
        u_rel  = self.p0_u[idx_g] - uc
        v_rel  = self.p0_v[idx_g] - vc
        w_rel  = self.p0_w[idx_g] - wc
        v2_rel = u_rel**2 + v_rel**2 + w_rel**2
        
        # Kinetic and gravitational potential energy.
        e_kin  = 0.5 * v2_rel * self.p0_mass[idx_g]
        e_grav = np.divide((self.G * mc * self.p0_mass[idx_g]), r)
        
        # Is the gas cell gravitationally bound?
        is_bound = np.greater(e_grav, e_kin)
        
        return is_bound
    
    # Get orbital elements of gas particles within certain radial distance of a
    # system with mass mc, (xc, yc, zc), (uc, vc, wc) (assuming [max_rad] = AU).
    def get_orbital_elements_from_system(self, max_rad, mc, xc, yc, zc, uc, vc, wc, gas_id):
        
        # Unit conversions: [L] = L_code_units 
        cm_to_m  = 1.0e-2
        cm_to_au = 6.6845871226706e-14
        m_to_au  = 6.6845871226706e-12
        
        # Index corresponding to chosen gas id.
        idx_g = np.where(self.p0_ids == gas_id)[0][0]
        
        # Relative position.
        x_rel = self.p0_x[idx_g] - xc
        y_rel = self.p0_y[idx_g] - yc
        z_rel = self.p0_z[idx_g] - zc
        r     = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        # Relative velocity.
        u_rel  = self.p0_u[idx_g] - uc
        v_rel  = self.p0_v[idx_g] - vc
        w_rel  = self.p0_w[idx_g] - wc
        v2_rel = u_rel**2 + v_rel**2 + w_rel**2
        
        # Kinetic and gravitational potential energy.
        e_kin  = 0.5 * v2_rel * self.p0_mass[idx_g]
        e_grav = np.divide((self.G * mc * self.p0_mass[idx_g]), r)
        
        # Is the gas cell gravitationally bound?
        is_bound = np.greater(e_grav, e_kin)
        
        # Is the gas cell within the maximum range specified?
        in_range = np.less(r * self.length * cm_to_au, max_rad)
        
        # Does the gas cell satisfy both criteria? If not, return.
        if not (is_bound or in_range):  # (not A) and (not B) == not (A or B)
            print('Gas particle either not gravitationally bound or else out of range.')
            return -1.0, -1.0, -1.0
            
        # Else, calculate orbital elements.
        x = np.asarray([x_rel, y_rel, z_rel]) * self.length   * cm_to_m  # [m]
        v = np.asarray([u_rel, v_rel, w_rel]) * self.velocity * cm_to_m  # [m/s]
    
        # Unit conversion factors.
        convert_1 = self.length * cm_to_m; convert_2 = self.velocity * cm_to_m
        
        # Standard gravitational parameter.
        mu = self.G * mc * convert_1 * convert_2**2
    
        a, e, j, raan, arg_pe, f = elements_from_state_vector(x, v, mu)
    
        # Report values in AU.
        a *= m_to_au
        r_ap = (1.0 + e) * a
            
        return a, e, r_ap
    
    # Center of mass of two-body system.
    def two_body_center_of_mass(self, m1, x1, v1, m2, x2, v2):
        
        m = m1 + m2
        cm_x = (m1 * x1[0] + m2 * x2[0]) / m
        cm_y = (m1 * x1[1] + m2 * x2[1]) / m
        cm_z = (m1 * x1[2] + m2 * x2[2]) / m
    
        cm_u = (m1 * v1[0] + m2 * v2[0]) / m
        cm_v = (m1 * v1[1] + m2 * v2[1]) / m
        cm_w = (m1 * v1[2] + m2 * v2[2]) / m
    
        return m, np.asarray([cm_x, cm_y, cm_z]), np.asarray([cm_u, cm_v, cm_w])
    
    # Compute center of mass of a sink-gas particle system given the sink particle ID
    # and a list of gas particle IDs.
    def system_center_of_mass(self, sink_ids, gas_ids):
        
        # Only calculate if star particle exists.
        if self.num_p5 == 0:
            return
        
        # Include only sink/gas particles specified.
        idx_s = np.isin(self.p5_ids, sink_ids)
        idx_g = np.isin(self.p0_ids, gas_ids)
        
        # Sink particle properties.
        m_sink = self.p5_mass[idx_s]
        x_sink = self.p5_x[idx_s]; y_sink = self.p5_y[idx_s]; z_sink = self.p5_z[idx_s]
        u_sink = self.p5_u[idx_s]; v_sink = self.p5_v[idx_s]; w_sink = self.p5_w[idx_s]
        
        # Gas particle properties.
        m_gas = self.p0_mass[idx_g]
        x_gas = self.p0_x[idx_g]; y_gas = self.p0_y[idx_g]; z_gas = self.p0_z[idx_g]
        u_gas = self.p0_u[idx_g]; v_gas = self.p0_v[idx_g]; w_gas = self.p0_w[idx_g]
        
        # Total mass of the system.
        m_total = np.sum(m_sink) + np.sum(m_gas)
        
        # Center of mass.
        x_cm = (np.sum(np.multiply(m_gas, x_gas)) + np.sum(np.multiply(m_sink, x_sink))) / m_total
        y_cm = (np.sum(np.multiply(m_gas, y_gas)) + np.sum(np.multiply(m_sink, y_sink))) / m_total
        z_cm = (np.sum(np.multiply(m_gas, z_gas)) + np.sum(np.multiply(m_sink, z_sink))) / m_total
        
        u_cm = (np.sum(np.multiply(m_gas, u_gas)) + np.sum(np.multiply(m_sink, u_sink))) / m_total
        v_cm = (np.sum(np.multiply(m_gas, v_gas)) + np.sum(np.multiply(m_sink, v_sink))) / m_total
        w_cm = (np.sum(np.multiply(m_gas, w_gas)) + np.sum(np.multiply(m_sink, w_sink))) / m_total
        
        cm_x = np.asarray([x_cm, y_cm, z_cm])
        cm_v = np.asarray([u_cm, v_cm, w_cm])
        
        return m_total, cm_x, cm_v
    
    # Compute center of mass of a sink-gas particle system given the sink particle ID
    # and a list of gas particle IDs.
    def gas_system_center_of_mass(self, gas_ids):
        
        # Include only gas particles specified.
        idx_g = np.isin(self.p0_ids, gas_ids)
        
        # Gas particle properties.
        m_gas = self.p0_mass[idx_g]
        x_gas = self.p0_x[idx_g]; y_gas = self.p0_y[idx_g]; z_gas = self.p0_z[idx_g]
        u_gas = self.p0_u[idx_g]; v_gas = self.p0_v[idx_g]; w_gas = self.p0_w[idx_g]
        
        # Total mass of the system.
        m_total = np.sum(m_gas)
        
        # Center of mass.
        x_cm = np.sum(np.multiply(m_gas, x_gas)) / m_total
        y_cm = np.sum(np.multiply(m_gas, y_gas)) / m_total
        z_cm = np.sum(np.multiply(m_gas, z_gas)) / m_total
        
        u_cm = np.sum(np.multiply(m_gas, u_gas)) / m_total
        v_cm = np.sum(np.multiply(m_gas, v_gas)) / m_total
        w_cm = np.sum(np.multiply(m_gas, w_gas)) / m_total
        
        cm_x = np.asarray([x_cm, y_cm, z_cm])
        cm_v = np.asarray([u_cm, v_cm, w_cm])
        
        return m_total, cm_x, cm_v
    
    # Return array of sink particle IDs sorted by distance from the sink particle
    # specified by the given sink particle ID.
    def sort_sinks_by_distance_from_sink(self, sink_id):
        
        # Need at least two sink particles.
        if self.num_p5 < 2:
            return
        
        # Index corresponding to chosen sink id.
        idx = np.where(self.p5_ids == sink_id)[0][0]
        
        # Particle IDs of other sink particles.
        id_rest = np.delete(self.p5_ids, idx, 0)
        
        # Relative coordinates of other sink particles.
        x_rest = (np.delete(self.p5_x, idx, 0) - self.p5_x[idx])
        y_rest = (np.delete(self.p5_y, idx, 0) - self.p5_y[idx])
        z_rest = (np.delete(self.p5_z, idx, 0) - self.p5_z[idx])
        
        # Distance from chosen sink particle.
        r = np.sqrt(x_rest**2 + y_rest**2 + z_rest**2)
        
        # Sort r, sink IDs, in ascending order.
        idx_sort       = np.argsort(r)
        id_rest_sorted = id_rest[idx_sort]
        
        # Also return distances in sorted order.
        r_sorted = r[idx_sort]
        
        return id_rest_sorted, idx_sort, r_sorted
    
    # Return array of gas particle IDs sorted by distance from the sink particle
    # specified by the given sink particle ID.
    def sort_gas_by_distance_from_sink(self, sink_id):
        
        # Only calculate if star particle exists.
        if self.num_p5 == 0:
            return
        
        # Index corresponding to chosen sink id.
        idx = np.where(self.p5_ids == sink_id)[0][0]
        
        # Relative coordinates of gas particles.
        x_rel = self.p0_x - self.p5_x[idx]
        y_rel = self.p0_y - self.p5_y[idx]
        z_rel = self.p0_z - self.p5_z[idx]
        
        # Distance from chosen sink particle.
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        # Sort r, gas particle IDs, in ascending order.
        idx_sort = np.argsort(r)
        id_gas_sorted = self.p0_ids[idx_sort]
        
        # Also return distances in sorted order.
        r_sorted = r[idx_sort]
        
        return id_gas_sorted, idx_sort, r_sorted
      
    '''
    # Return array of gas particle IDs sorted by distance from the
    # point specified by (x0, y0, z0) (e.g., center of mass).
    def sort_gas_by_distance_from_point(self, x0, y0, z0, mask):
        
        gas_ids = self.p0_ids[mask]
        
        # Relative coordinates of gas particles.
        x_rel = self.p0_x[mask] - x0
        y_rel = self.p0_y[mask] - y0
        z_rel = self.p0_z[mask] - z0
        r     = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        # Sort r, gas particle IDs, in ascending order.
        idx_sort = np.argsort(r)
        id_gas_sorted = gas_ids[idx_sort]
        
        # Also return distances in sorted order.
        r_sorted = r[idx_sort]
        
        return id_gas_sorted, idx_sort, r_sorted
    '''
    
    # Return array of gas particle IDs sorted by distance from the
    # point specified by (x0, y0, z0) (e.g., center of mass).
    def sort_gas_by_distance_from_point(self, x0, y0, z0, mask):
        
        gas_ids = self.p0_ids
        
        # Relative coordinates of gas particles.
        x_rel = self.p0_x - x0
        y_rel = self.p0_y - y0
        z_rel = self.p0_z - z0
        r     = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        # Sort by increasing distance.
        idx_sort = np.argsort(r)
        
        # Return gas particle IDs, sorted indices, mask, distance in sorted order.
        id_gas_sorted = gas_ids[idx_sort]
        mask_sorted   = mask[idx_sort]
        r_sorted      = r[idx_sort]
        
        return id_gas_sorted, idx_sort, mask_sorted, r_sorted
    
    # Return partcle mask sorted 
    def sort_indices_by_distance_from_point(self, x0, y0, z0):
        
        gas_ids = self.p0_ids
        
        # Relative coordinates of gas particles.
        x_rel = self.p0_x - x0
        y_rel = self.p0_y - y0
        z_rel = self.p0_z - z0
        r     = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        # Sort by increasing distance.
        idx_sorted = np.argsort(r)
        r_sorted   = r[idx_sorted]
        
        return idx_sorted, r_sorted
        
