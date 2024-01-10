import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

hb = pd.read_csv('hb.csv')
#pi = pd.read_csv('pi.csv')
#CHpi = pd.read_csv('CHpi.csv')

# plot freq of each interaction type in descending order - all sims combined
hb_pf = hb["type"].value_counts()
#pi_pf = pi["type"].value_counts()
#CHpi_pf = CHpi["type"].value_counts()
df = pd.concat([hb_pf], axis=0, ignore_index=False)
my_colors = list('bbbbbbbbbbbbbbbbbbggg')
df.sort_values(ascending=False).plot(kind='bar', color=my_colors)
plt.ylabel('Frequency')
plt.xlabel('Interaction type')
plt.tight_layout()
plt.show()

# plot hb count vs time or hist of all trajectory - all sims combined
#hb_count = [len(ts["type"]) for ts in hb]
#t = [t * timeslice for t in range(len(hb_count))]
#plt.plot(t, hb_count)
#plt.ylabel('Number of H-bonds')
#plt.xlabel('Time')
#plt.show()

# plot no. of offset and face-face pi stacking - all sims combined
#pi_off = pi_traj["is offset"].value_counts()
#pi_off.plot.bar(color="green")
#plt.show()


## Plot average, min and max frequency of each interaction type - each sim
#
#dfs = []
#
## get interaction counts for each amorphous cell
#for i in range(0,20):
#    hbname = 'each20/hb' + str(i) + '.csv'
#    piname = 'each20/pi' + str(i) + '.csv'
#    CHpiname = 'each20/CHpi' + str(i) + '.csv'
#    
#    hb = pd.read_csv(hbname)
#    pi = pd.read_csv(piname)
#    CHpi = pd.read_csv(CHpiname)
#
#    # plot freq of each interaction type in descending order
#    hb_pf = hb["type"].value_counts()
#    pi_pf = pi["type"].value_counts()
#    CHpi_pf = CHpi["type"].value_counts()
#    df = pd.concat([hb_pf, pi_pf, CHpi_pf], axis=0, ignore_index=False)
#    df.sort_values(ascending=False)
#    dfs.append(df)
#
## convert value counts from series to dataframes
#for i in range(0,20):
#    dfs[i] = pd.DataFrame(dfs[i]).reset_index()
#    dfs[i].columns = ["type", "frequency"]
#
## get mean, min and max counts for each interaction type
#dfmerge = reduce(lambda df1,df2: pd.merge(df1,df2,on="type",how="outer"), dfs)
#dfmerge["average"] = dfmerge.mean(axis=1)
#dfmerge["min"] = dfmerge.min(axis=1)
#dfmerge["max"] = dfmerge.max(axis=1)
#dfmerge.sort_values(ascending=False, by="average", inplace=True)
#dfmerge.reset_index(drop=True, inplace=True)
#
## make dictionary for error bars
#errors = {}
#for i in range(len(dfmerge)):
#    key = dfmerge["average"][i]
#    min = dfmerge["min"][i]
#    max = dfmerge["max"][i]
#    errors[key] = {"max": max, "min": min}
#
## plot the average counts for each type, with min/max error bars
#fig, ax = plt.subplots()
#dfmerge.plot.bar(x="type", y="average", ax=ax)
#for p in ax.patches:
#    x = p.get_x()
#    w = p.get_width()
#    h = p.get_height()
#    min_y = errors[h]['min']
#    max_y = errors[h]['max']
#    plt.vlines(x + w/2, min_y, max_y, color='k')
#plt.show()
#
