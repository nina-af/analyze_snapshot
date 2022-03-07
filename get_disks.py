# Get disks around sink 3, binary system (sinks 1 and 2), and triple disks
# for each snapshot.

import numpy as np
import os
import sys

import analyze_snapshot as snap

# Unit conversions.
unit_length_in_cm         = 3.085678e18  # 1 pc
unit_mass_in_g            = 1.989e33     # 1 solar mass
unit_velocity_in_cm_per_s = 1.0e2        # 1 m/s
unit_B_in_gauss           = 1.0e4        # 1 Tesla

unit_time_in_s  = unit_length_in_cm / unit_velocity_in_cm_per_s
unit_time_in_yr = unit_time_in_s / (3600.0 * 24 * 365)

cm_to_m = 1.0e-2
cm_to_au = 6.6845871226706e-14
pc_to_m = 3.08567758128e16
pc_to_au = 206264.80624538
m_to_au = 6.6845871226706e-12

# Input/output directories.
basedir   = '/work2/08381/nina_af/stampede2/gizmo_imf/output_A05_B13_nonideal_restart_273/'
outputdir = '/work2/08381/nina_af/stampede2/gizmo_imf/nonideal_disks/'
# Snapshots 273_000 through 273_040.

# Sinks 127830 (sink_1), 39456 (sink_2), and 17115 (sink_3).
sink_id_1 = 127830; sink_id_2 = 39456; sink_id_3 = 17115

# Add to disk if e < 0.3 and r_api < max_rad.
def in_disk(e, r_ap, max_rad):
    if e == -1.0 and r_ap == -1.0:
        return False
    elif e < 0.3 and r_ap < max_rad:
        return True
    else:
        return False
    
# Get gas particle disk around system with sys_m, sys_x, and sys_v.
# Mask: gas particles to include in examination.
def get_system_disk(s, sm, sx, sv, max_rad, mask, verbose=False):
    
    sys_m = sm; sys_x = sx; sys_v = sv
    
    r_ter = max_rad / pc_to_au  # Terminate disk extraction at r_ter.
    
    # Sort all gas particles by distance to system center of mass.
    idx_sorted, r_sorted = s.sort_indices_by_distance_from_point(sys_x[0], sys_x[1], sys_x[2])
    ids_sorted  = s.p0_ids[idx_sorted]
    
    # Consider only those particles which are included in the mask. To speed up loops, exclude 
    # all particles beyond r_ter from the mask.
    max_idx = np.argmax(r_sorted > r_ter)
    mask[idx_sorted[max_idx:]] = False
    mask_sorted = mask[idx_sorted]

    g_sort = ids_sorted[mask_sorted]  # Sorted particle IDs.
    g_idx  = idx_sorted[mask_sorted]  # Sorted particle indices.
    
    print('Number of gas particles to loop over: {0:d}'.format(len(g_sort)))

    # Track particle IDs of gas particles belonging / NOT belonging to disk, as well as IDs of
    # previously-examined gas particles.
    disk_ids = []; not_disk_ids = []; examined_ids = []

    # Loop over gas particles until r_ter is reached.
    k = 0; num_g = len(g_sort)
    while k < num_g:
    
        sys_m_new, sys_x_new, sys_v_new = sys_m, sys_x, sys_v
    
        p_id  = g_sort[0]  # Particle ID of nearest gas particle.
        p_idx = g_idx[0]   # Particle index in unsorted array.
        
        # Particle distance from system center of mass.
        p_r  = s.get_distance_from_point(sys_x_new[0], sys_x_new[1], sys_x_new[2], p_id) 
    
        if verbose:
            print('Current particle: {0:d} [{1:d}] (r = {2:.8e})'.format(p_id, p_idx, p_r))
            #print('Next particle:    {0:d} [{1:d}] (r = {2:.8e})'.format(g_sort[1], g_idx[1], p_r2))
    
        # Check exit condition.
        if p_r > r_ter:
            print('TERMINATING LOOP: particle {0:d} at r = {1:.3e} (r_ter = {2:.3e})'.format(p_id, p_r, r_ter))
            break
        
        # Check if gas particle should be added to the disk.
        a, e, r_ap  = s.get_orbital_elements_from_system(max_rad, sys_m, sys_x[0], sys_x[1], sys_x[2], 
                                                         sys_v[0], sys_v[1], sys_v[2], p_id)
            
        if verbose:
            print('a = {0:.3f}\te = {1:.3f}\tr_ap={2:.3f}'.format(a, e, r_ap))
            
        add_to_disk = in_disk(e, r_ap, max_rad)
    
        # Add gas particle ID to relevant lists.
        if add_to_disk:
            if verbose:
                print('Particle {0:d} in disk.'.format(p_id))
                
            pm = s.p0_mass[p_idx]
            px = np.asarray([s.p0_x[p_idx], s.p0_y[p_idx], s.p0_z[p_idx]])
            pv = np.asarray([s.p0_u[p_idx], s.p0_v[p_idx], s.p0_w[p_idx]])
                
            # Update disk-star system by adding gas particle.  
            sys_m_new, sys_x_new, sys_v_new = s.two_body_center_of_mass(sys_m, sys_x, sys_v, pm, px, pv)
            disk_ids.append(p_id)
            
        else:
            if verbose:
                print('Particle {0:d} NOT in disk.'.format(p_id))
            not_disk_ids.append(p_id)
            
        examined_ids.append(p_id)
    
        if verbose:
            print('Current system: m = {0}'.format(sys_m_new))
            print(sys_x_new); print(sys_v_new)
    
        # Update gas particle mask (True if NOT yet examined; resorted to match original data indexing).
        mask[p_idx] = False
        #print('# of non-zero elements in mask: {0:d}'.format(np.count_nonzero(mask)))
        
        # Update the system. 
        sys_m, sys_x, sys_v = sys_m_new, sys_x_new, sys_v_new
    
        # Resort remaining gas particles in order of distance from new system center of mass.
        idx_sorted, r_sorted = s.sort_indices_by_distance_from_point(sys_x[0], sys_x[1], sys_x[2])
        ids_sorted  = s.p0_ids[idx_sorted]
        mask_sorted = mask[idx_sorted]
   
        g_sort = ids_sorted[mask_sorted]  # Sorted particle IDs.
        g_idx  = idx_sorted[mask_sorted]  # Sorted particle indices.
        
        if verbose:
            print('----------------------------------------------\n')
            
        if k % 1000 == 0:
            print(k, sys_m, p_r * pc_to_au, r_ter * pc_to_au, len(g_sort), len(disk_ids))
        
        k += 1
        
    print('k = {0:d}: final particle: r = {1:.8e} pc = {2:.8e} AU'.format(k, p_r, p_r * pc_to_au))
        
    d_ids  = np.asarray(disk_ids, dtype=int)
    nd_ids = np.asarray(not_disk_ids, dtype=int)
    ex_ids = np.asarray(examined_ids, dtype=int)
        
    return d_ids, nd_ids, ex_ids


# Loop over all snapshots.
for i in range(41):
    
    print('\n SNAPSHOT_273_{0:03d}: getting disks...'.format(i))
    
    fname = os.path.join(basedir, 'snapshot_273_{0:03d}'.format(i))
    
    fname_3 = os.path.join(outputdir, 'snapshot_273_{0:03d}_disk_ids_3.txt'.format(i))
    fname_b = os.path.join(outputdir, 'snapshot_273_{0:03d}_disk_ids_b.txt'.format(i))
    fname_t = os.path.join(outputdir, 'snapshot_273_{0:03d}_disk_ids_t.txt'.format(i))
    
    s = snap.analyze_snapshot(fname)
    
    idx_s1 = np.where(s.p5_ids == sink_id_1)[0][0]
    idx_s2 = np.where(s.p5_ids == sink_id_2)[0][0]
    idx_s3 = np.where(s.p5_ids == sink_id_3)[0][0]
    
    ms1 = s.p5_mass[idx_s1]
    ms2 = s.p5_mass[idx_s2]
    ms3 = s.p5_mass[idx_s3]
    
    xs1 = s.p5_x[idx_s1]; ys1 = s.p5_y[idx_s1]; zs1 = s.p5_z[idx_s1]
    xs2 = s.p5_x[idx_s2]; ys2 = s.p5_y[idx_s2]; zs2 = s.p5_z[idx_s2]
    xs3 = s.p5_x[idx_s3]; ys3 = s.p5_y[idx_s3]; zs3 = s.p5_z[idx_s3]
    
    us1 = s.p5_u[idx_s1]; vs1 = s.p5_v[idx_s1]; ws1 = s.p5_w[idx_s1]
    us2 = s.p5_u[idx_s2]; vs2 = s.p5_v[idx_s2]; ws2 = s.p5_w[idx_s2]
    us3 = s.p5_u[idx_s3]; vs3 = s.p5_v[idx_s3]; ws3 = s.p5_w[idx_s3]
    
    pos_s1 = np.asarray([xs1, ys1, zs1]); vel_s1 = np.asarray([us1, vs1, ws1])
    pos_s2 = np.asarray([xs2, ys2, zs2]); vel_s2 = np.asarray([us2, vs2, ws2])
    pos_s3 = np.asarray([xs3, ys3, zs3]); vel_s3 = np.asarray([us3, vs3, ws3])
    
    # Binary sink system center of mass and coordinates.
    bin_m, bin_x, bin_v = s.two_body_center_of_mass(ms1, pos_s1, vel_s1, 
                                                    ms2, pos_s2, vel_s2)
    
    # Triple sink system center of mass and coordinates.
    trip_m, trip_x, trip_v = s.two_body_center_of_mass(bin_m, bin_x, bin_v, 
                                                       ms3, pos_s3, vel_s3)
    
    r_max0 = np.sqrt(np.sum((bin_x - pos_s3)**2)) * pc_to_au  # AU
    r_max = np.minimum(r_max0, 2000.0)
    mask  = np.full(len(s.p0_ids), True, dtype=bool)
    
    # Get disk around sink 3.
    print('Getting disk around sink 3...')
    disk_ids_3, not_disk_3, ex_disk_3 = get_system_disk(s, ms3, pos_s3, vel_s3, r_max, mask)
    print('Done. {0:d} particles in disk'.format(len(disk_ids_3)))
    np.savetxt(fname_3, disk_ids_3, fmt='%-7d')
    
    # Get circumbinary disk.
    print('Getting circumbinary disk...')
    mask_cb = np.isin(s.p0_ids, disk_ids_3, invert=True)     # All gas particles not in disk 3.
    disk_ids_cb, not_disk_cb, ex_disk_cb = get_system_disk(s, bin_m, bin_x, bin_v, r_max, mask_cb)
    print('Done. {0:d} particles in disk'.format(len(disk_ids_cb)))
    np.savetxt(fname_b, disk_ids_cb, fmt='%-7d')
    
    # Get circum-multiple disk around all 3 sinks.
    print('Getting circum-multiple disk...')
    all_disk_ids = np.union1d(disk_ids_3, disk_ids_cb)
    mask_sys = np.isin(s.p0_ids, all_disk_ids, invert=True)
    
    if len(all_disk_ids) > 0:
        gas_sys_m, gas_sys_x, gas_sys_v = s.gas_system_center_of_mass(all_disk_ids)
        sys_m, sys_x, sys_v = s.two_body_center_of_mass(trip_m, trip_x, trip_v, 
                                                        gas_sys_m, gas_sys_x, gas_sys_v)
    else:
        sys_m, sys_x, sys_v = trip_m, trip_x, trip_v
    
    r_max = 2000.0  # AU
    disk_ids_trip, not_disk_trip, ex_disk_trip = get_system_disk(s, sys_m, sys_x, sys_v, r_max, mask_sys)
    print('Done. {0:d} particles in disk'.format(len(disk_ids_trip)))
    np.savetxt(fname_t, disk_ids_trip, fmt='%-7d')
