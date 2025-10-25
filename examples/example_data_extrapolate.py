import numpy as np
import matplotlib.pyplot as plt
import resummation as res

# Example running the resummation model for baryon fractions drawn from data with Gaussian uncertainties, extrapolating R_500,crit to R_200,mean, assuming a Planck cosmology

# Read some mock data with uncertainties
mockdata=np.genfromtxt(f"mockdata.csv",delimiter=",",skip_header=1)
numdata=len(mockdata[:,0])

# Set up the samples of baryon fractions within R_500,crit
numsamples=1000
fb500c_samples=mockdata[None,:,1]+mockdata[None,:,2]*np.random.randn(numsamples,numdata)
fb500c_samples[fb500c_samples<0]=1e-3
fb500c_samples[fb500c_samples>0.25]=0.25

# Initialize the model with the cosmology, redshift, overdensity regions and DMO spectra to use
model=res.Resummation(cosmology="PLANCK",z=0,regions=["200_mean","500_crit"],highresspectra=0)

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

# Run the resummation model for each random sample of baryon fractions within R_500,crit, extrapolated to R_200,mean (see section 3.5 of VD25)
for i in np.arange(numsamples):
    model.set_fret_from_fb(mockdata[:,0],fb500c_samples[i,:],region="500_crit",extrapolate500cto200m=True)
    kres,qqres=model.run_resummation()
    plt.plot(kres,qqres,color="purple",alpha=0.25)
model.set_fret_from_fb(mockdata[:,0],mockdata[:,1],region="500_crit",extrapolate500cto200m=True)
kres,qqres=model.run_resummation()
plt.plot(kres,qqres,color="black",label="Mean data prediction")
plt.plot([None],[None],color="purple",alpha=0.25,label="Samples")

# Finish the plot
plt.legend(frameon=False,loc="lower left",fontsize=legendfontsize)
plt.savefig(f"resummation_data_extrapolate.pdf",transparent=True,dpi=300)
plt.close()
