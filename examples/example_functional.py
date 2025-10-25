import numpy as np
import matplotlib.pyplot as plt
import resummation as res

# Example running the resummation model for baryon fractions derived from a functional form, using the L2p8_m9 power spectra

# Use a sigmoid (tanh) to fit baryon fractions as a function of M_500,crit
def fitting_function(m,a,b,c):
	return c+a*(0.5*(1+np.tanh(m-b))-1)

# Initialize the model with the cosmology, redshift, overdensity regions and DMO spectra to use
model=res.Resummation(cosmology="FIDUCIAL",z=0,regions=["200_mean","500_crit"],highresspectra=2)

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
plt.xlim(2e-2,10)
plt.ylim(0.75,1.15)
plt.plot([1e-2,3e1],[1,1],linestyle=':',color="black")

# Run the resummation model for two sets of parametrized baryon fractions and plot the result
m500c=np.linspace(11,16,51) # the model will interpolate/extrapolate to the centres of the DMO cross spectra mass bins
model.set_fret_from_fb(m500c,fitting_function(m500c,0.10,13.6,0.15),region="200_mean")
model.set_fret_from_fb(m500c,fitting_function(m500c,0.12,14.2,0.16),region="500_crit")
kres,qqres=model.run_resummation()
plt.plot(kres,qqres,label="Resummation output (200,mean + 500,crit)",color="red",linestyle="--")

# Finish the plot
plt.legend(frameon=False,loc="lower left",fontsize=legendfontsize)
plt.savefig(f"resummation_functional.pdf",transparent=True,dpi=300)
plt.close()
