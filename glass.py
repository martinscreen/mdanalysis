"""Runs amorphonalysis on all glasses in turn and compiles all data into one
set of files for output. Data can then be treated as one large simulation."""

import os
import amorphonalysis as am
import MDAnalysis as mda
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def main():

    rootdir = os.getcwd()

    all_hb = []
    all_pi = []
    all_CHpi = []

    # iterate over all subdirectories starting with "glass"
    for subdir, dirs, files in os.walk(rootdir):
        for dir in dirs:
            if str(dir[:5]) == "glass":
                
                system = os.path.dirname(__file__) + "/" + dir + "/"
                u = am.load_universe(system)
                timeslice = 1

                # selections for AZ1 hydrogen bonds
                hb_hydrogens = u.select_atoms('name H27')
                hb_acceptors = u.select_atoms('name O N N1 N3 N4 N5')

                # selections for AZ1 aromatic groups
                #tetrazole = 'name N1 N2 N3 N4 C23'
                #phen1 = 'name C22 C21 C20 C19 C18 C17' 
                #phen2 = 'name C16 C15 C14 C13 C12 C11' 

                # make dicts where key is resID, value is atomgroup for that resID
                #tetrazoles = u.select_atoms(tetrazole).groupby('resids')
                #phen1s = u.select_atoms(phen1).groupby('resids')
                #phen2s = u.select_atoms(phen2).groupby('resids')
                #ff_groups = [tetrazoles, phen1s, phen2s]
                #ff_names = ["Tetra", "Phen1", "Phen2"]

                # selections for CH-pi hydrogens and acceptors:
                #CHpi_hydrogens = u.select_atoms('type ha or type h4').groupby('resids')
                #ind_6 = u.select_atoms('name C C3 C4 C5 C6 C7').groupby('resids')
                #ind_5 = u.select_atoms('name C C1 C2 C3 N').groupby('resids')
                #CHpi_acceptors = [ind_6, ind_5, poi_pyrimidines, e3_rings]
                #CHpi_names = ["ind6", "ind5", "pyr", "e3r"]

                # iterate over trajectory and gather data
                hb, pi, CHpi = [], [], []
                for ts in tqdm(u.trajectory[::timeslice]):

                    box = u.dimensions

                    # identify hydrogen-bonds for this timestep
                    ts_hbdata = am.find_hbonds(hb_hydrogens, hb_acceptors, box)
                    if type(ts_hbdata) == pd.core.frame.DataFrame:
                        ts_hbdata['timestep'] = str(ts.time)
                        hb.append(ts_hbdata)

                    # identify aromatic stacking for this timestep
                    #ts_pidata = am.find_stacking(ff_groups, ff_names, box)
                    #if type(ts_pidata) == pd.core.frame.DataFrame:
                    #    ts_pidata['timestep'] = str(ts.time)
                    #    pi.append(ts_pidata)

                    # identify CH-pi interactions for this timestep
                    #ts_CHpidata = am.find_CHpi(
                    #    CHpi_hydrogens, CHpi_acceptors, CHpi_names, box
                    #)
                    #if type(ts_CHpidata) == pd.core.frame.DataFrame:
                    #    ts_CHpidata['timestep'] = str(ts.time)
                    #    CHpi.append(ts_CHpidata)
                
                all_hb.append(hb)
                #all_pi.append(pi)
                #all_CHpi.append(CHpi)
    

    # concatenate dataframes across all simulations and timesteps for plotting
    hb_data = am.assemble_sims(all_hb)
    #pi_data = am.assemble_sims(all_pi)
    #CHpi_data = am.assemble_sims(all_CHpi)

    # write dataframes to files
    hb_data.to_csv('hb.csv', index=False)
    #pi_data.to_csv('pi.csv', index=False)
    #CHpi_data.to_csv('CHpi.csv', index=False)


if __name__ == "__main__":
    main()
