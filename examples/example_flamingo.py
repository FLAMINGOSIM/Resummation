import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import inspect
from flamingofunctions import getflamingolegend,getflamingocosmology
import resummation as res

# Example running the resummation model for a set of FLAMINGO simulations, including stellar fractions

sims=["HYDRO_FIDUCIAL","HYDRO_WEAK_AGN","HYDRO_STRONG_AGN","HYDRO_STRONGER_AGN","HYDRO_STRONGEST_AGN"]
dmosim="DMO_FIDUCIAL"
box="L1000N1800"

# Initialize the model with the cosmology, redshift, overdensity regions and DMO spectra to use
model=res.Resummation(cosmology="FIDUCIAL",z=0,regions=["200_mean","500_crit","2500_crit"],highresspectra=0)

# Set up the plot
fontsize=16
legendfontsize=14
plt.figure(figsize=(10,6))
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tick_params(axis='both',which='major',direction='inout',length=8)
plt.tick_params(axis='both',which='minor',direction='inout',length=4)
plt.xlabel(f"$k$ [$h$/Mpc]",fontsize=fontsize)
plt.ylabel(f"$P_{{\mathrm{{hydro}}}}(k)/P_{{\mathrm{{DMO}}}}(k)$",fontsize=fontsize)
plt.xscale("log")
plt.xlim(2e-2,25)
plt.ylim(0.75,1.15)
plt.plot([1e-2,3e1],[1,1],linestyle=':',color="black")

# Get the true suppression signal for L1_m9 and plot it
basedir=os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
spectrumfile=os.path.join(basedir,f"../resummation/spectra/{box}_{dmosim}_0077_autopower.json")
with open(spectrumfile,"r") as json_file:
    dmodata=json.load(json_file)
cosm=getflamingocosmology(dmosim)
k=np.array(dmodata["k_values"])/cosm["h"]
dmopower=np.array(dmodata["power_spectrum"])
spectrumfile=os.path.join(basedir,f"../resummation/spectra/{box}_HYDRO_FIDUCIAL_0077_autopower.json")
with open(spectrumfile,"r") as json_file:
    hydropower=np.array((json.load(json_file))["power_spectrum"])
dpp=hydropower/dmopower
legname,legcolor=getflamingolegend("HYDRO_FIDUCIAL",box="L1000N1800")
plt.plot(k,dpp,label=legname+" (true suppression)",color=legcolor)

# Run the resummation model for the given set of simulations
for sim in sims:
    fracfile=os.path.join(basedir,f"../resummation/fractions/fbar_{sim}_{box}_0077_HBT.json")
    with open(fracfile,"r") as f:
        fbar=json.load(f)
    fracfile=os.path.join(basedir,f"../resummation/fractions/fstar_{sim}_{box}_0077_HBT.json")
    with open(fracfile,"r") as f:
        fstar=json.load(f)
    # Set the baryon fractions and stellar fractions, then call run_resummation()
    model.set_fret_from_fb(fbar["m500c"],fbar["fbar200m"],region="200_mean")
    model.set_fret_from_fb(fbar["m500c"],fbar["fbar500c"],region="500_crit")
    model.set_fret_from_fb(fbar["m500c"],fbar["fbar2500c"],region="2500_crit")
    model.set_fstar(fstar["m500c"],fstar["fstar2500c"],"2500_crit")
    kres,qqres=model.run_resummation()
    legname,legcolor=getflamingolegend(sim,box="L1000N1800")
    plt.plot(kres,qqres,label=legname+" (resummation)",color=legcolor,linestyle="--")

# Finish the plot
plt.legend(frameon=False,loc="lower left",ncol=np.clip(np.ceil(len(sims)/6),1,3),fontsize=legendfontsize)
plt.savefig(f"resummation_FLAMINGO.pdf",transparent=True,dpi=300)
plt.close()
