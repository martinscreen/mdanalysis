"""Main program defining the functions to use in glass analysis."""

import os
import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib import mdamath
from tqdm import tqdm
from MDAnalysis.lib import distances
import pandas as pd
import matplotlib.pyplot as plt


def load_universe(system):

    """Loads coords and trajectory into universe, then subsequently adds 
    topology info"""

    # load coords and traj into universe
    u = mda.Universe(system+'I1H_md1.gro', system+'md1.xtc')

    # load topology as separate universe then transfer info to main
    u_top = mda.Universe(system+'I1H.top', topology_format='ITP')
    u.add_TopologyAttr('types', u_top.atoms.types)
    u.add_TopologyAttr('charges', u_top.atoms.charges)
    u.add_TopologyAttr('masses', u_top.atoms.masses)
    u.add_bonds(u_top.bonds.to_indices())
    u.add_angles(u_top.angles.to_indices())
    u.add_dihedrals(u_top.dihedrals.to_indices())
    
    return u


def remove_duplicates(array):

    """Remove duplicates in an array arising from indices re-occuring in 
    reverse order"""
    
    for i in range(len(array)):
        for j in range(len(array)):
            if array[i][0] == array[j][1] and array[j][0] == array[i][1]:
                array[j][0] = 10000

    return np.delete(array, np.where(array[:,0] == 10000), axis=0) 


def find_normal(atomgroup):

    """Takes an atomgroup and finds plane between 3 atoms (assumes planar)"""
    
    a1, a2, a3 = atomgroup.atoms[:5:2]
    a1_pos = a1.position
    a2_pos = a2.position
    a3_pos = a3.position
    vector_a2 = a2_pos - a1_pos
    vector_a3 = a3_pos - a1_pos

    return mdamath.normal(vector_a2, vector_a3)


def find_angle(vector1, vector2):

    """Finds angle between two normal vectors"""

    angle = mdamath.angle(vector1, vector2) * 180 / np.pi
    
    # in case vectors are anti-parallel
    if angle >= 90:
        angle = 180 - angle
    
    return angle


def is_offset(norm1, norm2, com1, com2):

    """Determine if two aromatic rings are offset by >20 degrees"""

    com_vector = com2 - com1
    angle1 = mdamath.angle(norm1, com_vector) * 180 / np.pi
    angle2 = mdamath.angle(norm2, com_vector) * 180 / np.pi
    if angle1 >= 90:
        angle1 = 180 - angle1

    if angle2 >= 90:
        angle2 = 180 - angle2

    if angle1 >= 20 and angle2 >= 20:
        return True

    return False


def group_atomname(atom):
    
    """Ensures chemically similar atoms are combined by name"""

    if atom.name in ["N2", "N3"]:
        name = "Npyr"
    elif atom.name in ["N4","N5", "N6", "N7"]:
        name = "Ncyc"
    else:
        name = atom.name
    
    return name


def find_hbonds(hydrogens, acceptors, box, length_cutoff=3.5, 
                angle_cutoff=150):

    """Identifies hydrogen bonds based on cutoffs and returns indices of the
    atoms in each bond along with bond lengths"""

    donors = sum(h.bonded_atoms[0] for h in hydrogens)

    # get indices corresponding to acc & don pairs within cutoff length
    idx = distances.capped_distance(acceptors.positions, 
                                    donors.positions, 
                                    max_cutoff=length_cutoff, 
                                    box=box,
                                    return_distances=False
                                    )
    if len(idx) == 0:
        return

    # separate into indices for acceptors and donors identified
    acc_idx, hyd_idx = idx.T
    potential_acceptors = acceptors[acc_idx]
    potential_hydrogens = hydrogens[hyd_idx]
    # assuming donor atom is only atom bonded to the hydrogen:
    potential_donors = sum(h.bonded_atoms[0] for h in potential_hydrogens)

    # check angles
    angles = np.rad2deg(
        distances.calc_angles(potential_acceptors.positions, 
                              potential_hydrogens.positions, 
                              potential_donors.positions, box=box
                              )
    )
    angle_idx = np.where(angles >= angle_cutoff)
    if len(angle_idx[0]) == 0:
        return

    # identify hbonds within length and angle cutoffs
    hbond_acceptors = potential_acceptors[angle_idx]
    hbond_hydrogens = potential_hydrogens[angle_idx]
    hbond_donors = potential_donors[angle_idx]

    # initialise data arrays
    types, resid_is, resid_js, angles, lengths = [], [], [], [], []
    for i in range(len(hbond_acceptors)):
            acceptor = hbond_acceptors[i]
            hydrogen = hbond_hydrogens[i]
            donor = hbond_donors[i]
            acc_name = acceptor.name
            don_name = donor.name
            types.append(don_name + "H..." + acc_name)
            resid_is.append(acceptor.resid)
            resid_js.append(donor.resid)
            angles.append(np.rad2deg(
                distances.calc_angles(acceptor.position, hydrogen.position,
                                      donor.position, box=box)
            ))
            lengths.append(
                distances.calc_bonds(acceptor.position, donor.position, 
                                     box=box
                                     )
            )

    # assemble lists into pandas dataframe
    data = list(zip(types, resid_is, resid_js, angles, lengths))
    columns=['type', 'acceptor resid', 'donor resid', 'angle', 'length']
    df = pd.DataFrame(data, columns=columns)

    return df


def find_stacking(groups, groupnames, box, length_cutoff=4.1, angle_cutoff=30):

    """Identifies pi-stacking interactions in a timestep based on cutoffs and
    returns the group types in an interacting pair, their resids, the angle 
    and length between them."""

    # get list of COMs for each atomgroup
    coms = []
    for group in groups:
        group_coms = [i[1].atoms.center_of_mass(unwrap=True) 
                for i in group.items()
                ]
        coms.append(np.asarray(group_coms))
        
    # initialise data arrays
    types, resid_is, resid_js, angles, lengths, offsets = [], [], [], [], [], []

    # iterate over unique combinations of groups
    for i in range(len(groups)):
        for j in range(len(groups)):
            if i <= j:
            
                # get indices of pairs within length cutoff
                idx = distances.capped_distance(coms[i], coms[j], 
                                                max_cutoff=length_cutoff,
                                                min_cutoff=0.01,
                                                return_distances=False, 
                                                box=box
                                                )
                if len(idx) == 0:
                    continue

                # remove duplicates if looking at self-interactions
                if i == j:
                    idx = remove_duplicates(idx)
                
                # get angles between pairs
                angs = []
                for pair in idx:
                    # indices of COMs array = resid-1 in the atomgroup dict
                    resid_i = pair[0] + 1
                    resid_j = pair[1] + 1
                    norm_i = find_normal(groups[i][(resid_i)])
                    norm_j = find_normal(groups[j][(resid_j)])
                    angs.append(find_angle(norm_i, norm_j))

                stacked_idx = np.where(np.asarray(angs) <= angle_cutoff)
                if len(stacked_idx) == 0:
                    continue

                # get indices of COMs array for pairs within both cutoffs
                coms_idx = idx[stacked_idx]

                # iterate over these indices to save their data
                for k in range(len(coms_idx)):
                    resid_i = coms_idx[k][0] + 1
                    resid_j = coms_idx[k][1] + 1
                    com_i = coms[i][coms_idx[k][0]]
                    com_j = coms[j][coms_idx[k][1]]
                    angle = np.asarray(angs)[stacked_idx][k]
                    length = distances.calc_bonds(com_i, com_j, box=box)
                    offset = False

                    # if length >3.8, must be offset to continue
                    if length >= 3.8:
                        norm_i = find_normal(groups[i][(resid_i)])
                        norm_j = find_normal(groups[j][(resid_j)])
                        if not is_offset(norm_i, norm_j, com_i, com_j):
                            # this is a face-face interaction out of range
                            #print("Out of range: ", str(length))
                            continue
                        offset = True 

                    types.append(groupnames[i] + " " + groupnames[j])
                    resid_is.append(resid_i)
                    resid_js.append(resid_j)
                    angles.append(angle)
                    lengths.append(length)
                    offsets.append(offset)

    # assemble lists into pandas dataframe
    data = list(zip(types, resid_is, resid_js, angles, lengths, offsets))
    columns=['type', 'resID i', 'resID j', 'angle', 'length', 'is offset']
    df = pd.DataFrame(data, columns=columns)

    return df


def find_CHpi(hydrogens, acceptors, groupnames, box, length_cutoff=2.6,
              angle_cutoff=150):

    """Identifies CH-pi bonds based on cutoffs and returns indices of the
    atoms in each bond along with bond lengths"""

    # for atomgroup in acceptors, compute COMs
    coms = []
    for group in acceptors:
        group_coms = [i[1].atoms.center_of_mass(unwrap=True) 
                      for i in group.items()
                      ]
        coms.append(np.asarray(group_coms))

    # make separate lists of H atomgroups and positions
    hyds = []
    for resid, group in hydrogens.items():
        for atom in group.atoms:
            hyds.append(atom)

    hyd_pos = []
    for atom in hyds:
        hyd_pos.append(atom.position)
    hyd_pos = np.asarray(hyd_pos)

    # initialise data arrays
    types, resid_is, resid_js, angles, lengths = [], [], [], [], []

    for i in range(len(acceptors)):
        
        # get indices corresponding to acc & H pairs within cutoff length
        idx = distances.capped_distance(coms[i], 
                                        hyd_pos,
                                        max_cutoff=length_cutoff, 
                                        box=box,
                                        return_distances=False
                                        )
        if len(idx) == 0:
            return

        # check C-H-pi angle within angle cutoff
        angs = []
        for pair in idx:
            acc_com = coms[i][pair[0]]
            hydrogen = hyds[pair[1]]
            donor = hydrogen.bonded_atoms[0]
            angs.append(np.rad2deg(
                distances.calc_angles(acc_com, hydrogen.position,
                                      donor.position, box=box)
            ))

        # get indices of coms/hyds arrays for interacting atoms
        CHpi_idx = np.where(np.asarray(angs) >= angle_cutoff)
        if len(CHpi_idx) == 0:
            continue
        
        # iterate over these indices and save data into dataframe
        CHpis = idx[CHpi_idx]
        for k in range(len(CHpis)):
            acc_com = coms[i][CHpis[k][0]]
            hydrogen = hyds[CHpis[k][1]]
            angle = np.asarray(angs)[CHpi_idx][k]
            length = distances.calc_bonds(acc_com, hydrogen.position,
                                          box=box)

            types.append(hydrogen.name + " -> " + groupnames[i])
            resid_is.append(CHpis[k][0] + 1)
            resid_js.append(hydrogen.resid)
            angles.append(angle)
            lengths.append(length)

    # assemble lists into pandas dataframe
    data = list(zip(types, resid_is, resid_js, angles, lengths))
    columns=['type', 'acceptor resID', 'donor resID', 'angle', 'length']
    df = pd.DataFrame(data, columns=columns)

    return df


def assemble_list(sim):

    """Takes a list of data for each timestep of a set of simulations
    and concatenates into one pd.Dataframe"""

    frames = []
    for ts in sim:
        frames.append(ts)

    return pd.concat(frames)


def assemble_sims(data):

    """Takes a list of lists of data for each timestep of a set of simulations
    and concatenates into one pd.Dataframe"""

    frames = []
    for sim in data:
        for ts in sim:
            frames.append(ts)

    return pd.concat(frames)


def main():

    # specify path to system of choice
    system = os.path.dirname(__file__) + "/"
    u = load_universe(system)
    timeslice = 1

    # selections for AZ1 hydrogen bonds
    hb_hydrogens = u.select_atoms('name H')
    hb_acceptors = u.select_atoms('name O O1 O2 N1 N5 N6')

    # selections for AZ1 aromatic groups
    poi_indole = 'name C C1 C2 C3 C4 C5 C6 C7 N'
    poi_pyrimidine = 'name C11 C12 C13 C14 N2 N3' # also N4?
    e3_ring = 'name C31 C32 C33 C34 C35 C36' # also N7 C37 O N8?
    e3_formamide = 'name C40 C43 N9 O1 O2'

    # make dicts where key is resID, value is atomgroup for that resID
    poi_indoles = u.select_atoms(poi_indole).groupby('resids')
    poi_pyrimidines = u.select_atoms(poi_pyrimidine).groupby('resids')
    e3_rings = u.select_atoms(e3_ring).groupby('resids')
    e3_formamides = u.select_atoms(e3_formamide).groupby('resids')
    ff_groups = [poi_indoles, poi_pyrimidines, e3_rings, e3_formamides]
    ff_names = ["Ind", "Pyr", "E3R", "For"]

    # selections for CH-pi hydrogens and acceptors:
    CHpi_hydrogens = u.select_atoms('type ha or type h4').groupby('resids')
    ind_6 = u.select_atoms('name C C3 C4 C5 C6 C7').groupby('resids')
    ind_5 = u.select_atoms('name C C1 C2 C3 N').groupby('resids')
    CHpi_acceptors = [ind_6, ind_5, poi_pyrimidines, e3_rings]
    CHpi_names = ["ind6", "ind5", "pyr", "e3r"]

    # iterate over trajectory and gather data
    hb, pi, CHpi = [], [], []
    for ts in tqdm(u.trajectory[::timeslice]):
        
        box = u.dimensions
        
        # identify hydrogen-bonds for this timestep
        ts_hbdata = find_hbonds(hb_hydrogens, hb_acceptors, box)
        if type(ts_hbdata) == pd.core.frame.DataFrame:
            ts_hbdata['timestep'] = str(ts.time)
            hb.append(ts_hbdata)

        # identify aromatic stacking for this timestep
        ts_pidata = find_stacking(ff_groups, ff_names, box)
        if type(ts_pidata) == pd.core.frame.DataFrame:
            ts_pidata['timestep'] = str(ts.time)
            pi.append(ts_pidata)

        # identify CH-pi interactions for this timestep
        CHpi_data = find_CHpi(CHpi_hydrogens, CHpi_acceptors, CHpi_names, box)
        if type(CHpi_data) == pd.core.frame.DataFrame:
            CHpi_data['timestep'] = str(ts.time)
            CHpi.append(CHpi_data)

    # concatenate dataframes across all timesteps for plotting
    hb_data = assemble_list(hb)
    pi_data = assemble_list(pi)
    CHpi_data = assemble_list(CHpi)

    # write dataframes to files
    hb_data.to_csv('hb.csv', index=False)
    pi_data.to_csv('pi.csv', index=False)
    CHpi_data.to_csv('CHpi.csv', index=False)


if __name__ == "__main__":
    main()
