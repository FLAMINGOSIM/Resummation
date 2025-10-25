import numpy as np
import json
import os.path
import inspect
from scipy.ndimage import gaussian_filter1d
from .flamingofunctions import getflamingocosmology

class Resummation:
    """
    Resummation model for computing suppression signals from baryon fractions.

    This version of the resummation model is described in van Daalen et al. 
    (2025, VD25). Creating a new instance of Resummation initializes the model 
    to a set of DMO power spectra. See docstring of __init__ for 
    initialization parameters.

    After intializing the model, baryon fractions (or retained mass fractions 
    directly) should be set before calling run_resummation(). Optionally, 
    stellar fractions and a collective retained mass fraction for the combined 
    5xR500c regions can be set as well (see sections 2.2.3 and 3.6 of VD25, 
    respectively). The model does not need to be reinitialized in order to 
    provide predictions for new sets of baryon fractions: simply set new 
    baryon fractions on the existing object and call run_resummation() again.

    To set mean halo baryon fractions, use either set_fret_from_fb() (when 
    providing regular baryon fractions), or set_fret_from_fbc() (when 
    providing corrected baryon fractions). Alternatively, use set_fret() to 
    set retained mass fractions directly. See the documentation of these 
    functions for more information.
    """

    def __init__(self,cosmology="FIDUCIAL",z=0,regions=["200_mean","500_crit"],highresspectra=0):
        """
        Initialization of the resummation model. Here one sets the cosmology, 
        redshift, overdensity region(s) and DMO simulation to use.

        Parameters
        ----------

        cosmology: str, optional
            The cosmology to use for the DMO simulation, and to transform 
            baryon fractions to corrected baryon fractions. Options are 
            FIDUCIAL (default, equivalent to DESY3), PLANCK, 
            PLANCK_MID_NU_VARY, PLANCK_LARGE_NU_VARY, PLANCK_LARGE_NU_FIXED, 
            PLANCK_LARGER_NU_FIXED, and LOW_SIGMA8 (equivalent to LS8).

        z: float, optional
            The redshift at which to calculate suppression signals. Choose 
            from z=0 (default), z=0.5 or z=1.

        regions: list of str, optional
            A list of one or more regions for which baryon fractions will be 
            provided. Options are 50_crit, 200_mean, 500_crit, and 2500_crit. 
            The default is to use 200_mean and 500_crit.

        highresspectra: int, optional
            The type of DMO spectra to use. Value must be 0 (default), 1 or 2; 
            highresspectra=0 will use the L1_m9_DMO power spectra; 
            highresspectra=1 will use the L1_m8_DMO power spectra, and 
            highresspectra=2 will use the L2p8_m9_DMO power spectra. 
            Power spectra beyond L1_m9_DMO are only available for the FIDUCIAL 
            cosmology at z=0.

        """
        self.initialized=False
        self.basedir=os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        self.cosmology=cosmology
        self.cosm=getflamingocosmology(cosmology)
        if self.cosm is None:
            return
        self.dmosim=f"DMO_{cosmology}"
        if cosmology=="DESY3":
            self.dmosim="DMO_FIDUCIAL"
        elif cosmology=="LS8":
            self.dmosim="DMO_LOW_SIGMA8"
        if isinstance(z,list) or isinstance(z,np.ndarray):
            raise TypeError("Redshift z must be a scalar.")
        self.redshift=z
        if np.isclose(self.redshift,0):
            self.snap="0077"
        elif np.isclose(self.redshift,0.5):
            self.snap="0067"
        elif np.isclose(self.redshift,1):
            self.snap="0057"
        else:
            raise ValueError(f"No data is available for redshift z={z:.3f} (available: z=0, 0.5, 1).")
        self.box="L1000N1800"
        self.regions=regions
        self.SOdelta=[0.0]*len(regions)
        for i in np.arange(len(regions)):
            recognized=False
            if self.regions[i]=="5x500_crit":
                self.SOdelta[i]=-1
                self.regions[i]=""
                recognized=True
            elif "_crit" in self.regions[i]:
                factor=self.regions[i].split("_crit",1)[0]
                if factor.isdigit():
                    self.SOdelta[i]=float(factor)
                    recognized=True
            elif "_mean" in self.regions[i]:
                Ez=(self.cosm["Om"]-self.cosm["Onu"])*(1+self.cosm["Om"])**3+self.cosm["Onu"]*(1+self.redshift)**4+(1-self.cosm["Om"])
                factor=self.regions[i].split("_mean",1)[0]
                if factor.isdigit():
                    self.SOdelta[i]=float(factor)*self.cosm["Om"]*(1+self.redshift)**3/Ez
                    recognized=True
            if not recognized:
                print(f"Warning: region '{self.regions[i]}' is not supported and will be ignored.")
                self.regions[i]=""
        self.SOdelta,self.regions=zip(*sorted(zip([x for x,y in zip(self.SOdelta,self.regions) if y!=""],[x for x in self.regions if x!=""])))
        if len(self.regions)==0:
            raise ValueError("No valid overdensity regions.")
        self.highresspectra=highresspectra
        if self.highresspectra>0:
            if self.dmosim!="DMO_FIDUCIAL" or self.redshift>0:
                raise ValueError("High-res spectra are only available for the fiducial cosmology at z=0.")
            else:
                self.snap="00"+str(int(self.snap)+1)
                if self.highresspectra==1:
                    self.box="L1000N3600"
                else:
                    self.box="L2800N5040"
        # Note that the DMO cross spectra used are assumed to use these mass bin centres
        if np.isclose(self.redshift,0):
            if self.highresspectra==0:
                self.m500c_cen=np.array([10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25])
            elif self.highresspectra==1:
                self.m500c_cen=np.array([9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25])
            else:
                self.m500c_cen=np.array([11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25,15.75])
        else:
            if self.highresspectra==1:
                self.m500c_cen=np.array([10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75])
            else:
                self.m500c_cen=np.array([9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75])
        self.nbins=len(self.m500c_cen)
        self.__getmassfracs()
        self.__getpowerspectra()
        self.__getbiases()
        self.__getfretparams()
        self.fret=None
        self.fstar=None
        self.fret5x500_crit=None
        self.kmodel=None
        self.ptotmodel=None
        self.pnotmodel=None
        self.qqmodel=None
        self.initialized=True

    def __getmassfracs(self):
        self.massfrac={}
        for r in self.regions:
            #self.massfrac|={f"massfrac{r}": self.getmassfrac(region=r)} #|= for dicts is only supported in newer versions of Python
            self.massfrac.update({f"massfrac{r}": self.__getmassfrac(region=r)})
            mlen=len(self.massfrac[f"massfrac{r}"])
            if mlen!=self.nbins:
                raise RuntimeError(f"Mass fractions array do not all have the same size (there were {mlen} mass fractions for region {r} but expected {self.nbins}).")
        combinedmassfrac=self.__getmassfrac(region="5x500_crit_combined",noerror=True)
        if not np.isnan(combinedmassfrac[0]):
            if len(combinedmassfrac)==1:
                self.massfrac.update({f"massfrac5x500_crittot": combinedmassfrac[0]})

    def __getmassfrac(self,region="200_mean",noerror=False):
        mffile=os.path.join(self.basedir,f"fractions/massfrac_{self.dmosim}_{self.box}_{self.snap}_HBT_{region}.dat")
        if os.path.exists(mffile):
            return np.loadtxt(mffile,ndmin=1)
        else:
            if not noerror:
                raise ValueError(f"No mass fractions are available for this combination of input parameters (region={region}).")
            return np.array([np.nan])

    def __getpowerspectra(self):
        # Note that these power spectra are pre-multiplied with their mass fractions
        self.power={}
        spectrumfile=os.path.join(self.basedir,f"spectra/{self.box}_{self.dmosim}_{self.snap}_autopower.json")
        if not os.path.exists(spectrumfile):
            raise ValueError(f"No matter power spectrum is available for this combination of input parameters (cosmology={self.cosmology}, z={self.redshift}, highresspectra={self.highresspectra}).")
        with open(spectrumfile,"r") as json_file:
            autopower=json.load(json_file)
        self.power.update({"k": np.array(autopower["k_values"])/self.cosm["h"]})
        self.nk=len(self.power["k"])
        self.power.update({"ptot": np.array(autopower["power_spectrum"])})
        for r in self.regions:
            self.power.update({f"p{r}": np.zeros((self.nbins,self.nk),dtype=np.float64)})
        self.power.update({"p500haloes": np.zeros((self.nbins,self.nk),dtype=np.float64)})
        for bin in np.arange(self.nbins):
            for r in self.regions:
                spectrumfile=os.path.join(self.basedir,f"spectra/{self.box}_{self.dmosim}_{self.snap}_{r}_crosspower_HBT_bin_{bin}.json")
                if not os.path.exists(spectrumfile):
                    raise ValueError(f"No cross power spectrum is available for this combination of input parameters (region={r}, mass bin={bin}, cosmology={self.cosmology}, z={self.redshift}, highresspectra={self.highresspectra}).")
                with open(spectrumfile,"r") as f:
                    power=json.load(f)
                ktmp=np.array(power["k_values"])/self.cosm["h"]
                ptmp=np.array(power["power_spectrum"])
                self.power[f"p{r}"][bin,:]=10**np.interp(np.log10(self.power["k"]),np.log10(ktmp[ptmp>0]),np.log10(ptmp[ptmp>0]))
            spectrumfile=os.path.join(self.basedir,f"spectra/haloes_{self.box}_{self.dmosim}_{self.snap}_500_crit_crosspower_HBT_bin_{bin}.json")
            if not os.path.exists(spectrumfile):
                self.halospectra=False
            else:
                self.halospectra=True
                with open(spectrumfile,"r") as f:
                    power=json.load(f)
                ktmp=np.array(power["k_values"])/self.cosm["h"]
                ptmp=np.array(power["power_spectrum"])
                self.power["p500haloes"][bin,:]=10**np.interp(np.log10(self.power["k"]),np.log10(ktmp[ptmp>0]),np.log10(ptmp[ptmp>0]))
        spectrumfile=os.path.join(self.basedir,f"spectra/{self.box}_{self.dmosim}_{self.snap}_5x500_crit_crosspower_HBT_combined.json")
        if os.path.exists(spectrumfile):
            with open(spectrumfile,"r") as f:
                power=json.load(f)
            ktmp=np.array(power["k_values"])/self.cosm["h"]
            ptmp=np.array(power["power_spectrum"])
            self.power.update({"p5x500_crittot": 10**np.interp(np.log10(self.power["k"]),np.log10(ktmp[ptmp>0]),np.log10(ptmp[ptmp>0]))})
            self.power.update({"p5x500_critnot": self.power["ptot"]-self.power["p5x500_crittot"]})
        self.power.update({"poutertot": np.sum(self.power[f"p{self.regions[0]}"],axis=0)})
        self.power.update({"pouternot": (self.power["ptot"]-self.power["poutertot"])})
    
    def __getbiases(self):
        self.bias={}
        for r in self.regions:
            bias=np.zeros(self.nbins,dtype=np.float64)
            for bin in np.arange(self.nbins):
                bias[bin]=(np.mean(self.power[f"p{r}"][bin,1:7]/self.power["ptot"][1:7]))/self.massfrac[f"massfrac{r}"][bin]
            self.bias.update({f"bias{r}": bias})
        if "massfrac5x500_crittot" in self.massfrac and "p5x500_crittot" in self.power:
            bias5x500c=(np.mean(self.power[f"p5x500_crittot"][1:7]/self.power["ptot"][1:7]))/self.massfrac[f"massfrac5x500_crittot"]
            self.bias.update({"bias5x500_crittot": bias5x500c})

    def __getfretparams(self): # Assumes the universal fits to L2p8_m9 (see Figures 4 and B1)
        if np.isclose(self.redshift,0):
            psnap="0078"
        elif np.isclose(self.redshift,0.5):
            psnap="0068"
        else:
            psnap="0058"
        paramfile=os.path.join(self.basedir,f"fitparams/params_fret_vs_fbc_L2p8m9_{psnap}_ALL_bfixed_err_selectm500_noreject_highres_HBT_ownmass.json")
        if os.path.exists(paramfile):
            with open(paramfile,"r") as json_file:
                params=json.load(json_file)
            a0,a1,b,c0,c1,c2=params['params']
        else:
            raise RuntimeError(f"No universal fit parameter file found at {paramfile}!")
        self.fretparams={}
        for i in np.arange(len(self.regions)):
            self.fretparams.update({f"abc{self.regions[i]}": np.array([a0*(self.SOdelta[i]/100.0)**a1,b,(1+(self.SOdelta[i]/c0)**c1)**c2])})

    def get_fret_from_fbc(self,fbc,region):
        """
        Transforms corrected baryon fractions to retained mass fractions.

        Parameters
        ----------

        fbc: float or array_like
            The corrected baryon fractions to transform.

        region: str
            The corresponding overdensity region, which sets the 
            transformation parameters.

        Returns
        -------

        fret: np.array
            The retained mass fractions for the given corrected baryon 
            fractions and region.

        """
        if not self.initialized:
            raise RuntimeError("Resummation was initialized with errors; please create a new instance with compatible settings.")
        if region not in self.regions:
            print(f"Warning: baryon fractions for region {region} will be ignored, as this region was not set in initialization.")
            return np.array([np.nan]*len(fbc))
        return self.fretparams[f"abc{region}"][2]-self.fretparams[f"abc{region}"][1]*(1-np.clip(np.array(fbc),a_min=0.0,a_max=1.0))**self.fretparams[f"abc{region}"][0]
    
    def extrapolate500cto200m(self,fret500c):
        """
        Extrapolate the retained mass fractions within R500c to those within 
        R200m, using the relation in the left-hand side of Figure 10 of VD25.

        Parameters
        ----------

        fret500c: float or array_like
            The retained mass fractions measured inside R500c that will be 
            mapped to R200m.

        Returns
        -------

        fret200m: np.array
            The extrapolated retained mass fractions within R200m.

        """
        return 1.008/(1+np.exp(-15.514*(np.array(fret500c)-0.714))) # See left-hand side of Figure 10

    def set_fret_from_fbc(self,m500c,fbc,region=None,extrapolate500cto200m=False):
        """
        Transforms corrected baryon fractions to retained mass fractions. Use 
        either this function, set_fret_from_fb, or set_fret. This function has 
        no return value; retained mass fractions are set internally in the
        resummation object. Note that the retained mass fractions obtained are 
        linearly interpolated in log10(M500c) to the mass bins used for the 
        halo-matter cross spectra.

        Parameters
        ----------

        m500c: float or array_like
            Log10(M500c) halo masses for which corrected baryon fractions are 
            provided, in units of Msun (no h).

        fbc: type and length matching m500c, or dict
            The corrected baryon fractions for the given halo masses within 
            the radius corresponding to the given region. Based on these the 
            retained mass fractions will be determined. The corrected baryon 
            fractions can either be given for a single region, or as a 
            dictionary for all regions at once. If fbc is a dictionary, the 
            keys are expected to be of the form fbc200m, fbc500c, etc. In this 
            case, no region should be provided as a parameter.

        region: str, optional
            The corresponding overdensity region if fbc is a list or array of 
            floats. If fbc is given as a dict, this should be None.

        extrapolate500cto200m: bool, optional
            Whether corrected baryon fractions given for R500c should be 
            extrapolated to estimate retained mass fractions within R200m as 
            well, according to the relation of the left-hand side of Figure 10 
            of VD25. If this parameter is set to True for regions other than 
            R500c it will be ignored.

        """
        if not self.initialized:
            raise RuntimeError("Resummation was initialized with errors; please create a new instance with compatible settings.")
        if not isinstance(m500c,np.ndarray):
            m500c=np.array(m500c)
        if np.log10(np.min(m500c))>2:
            m500c=np.log10(m500c)
        if region is not None:
            fret=self.get_fret_from_fbc(fbc,region)
            if np.isnan(fret[0]):
                return
            self.set_fret(m500c,fret,region=region,extrapolate500cto200m=extrapolate500cto200m)
        else:
            if not isinstance(fbc,dict):
                raise TypeError("Corrected baryon fractions should be given in dictionary form, with region names as keys, or a region should be provided in the function call.")
            for r in self.regions:
                if not f"fbc{r}" in fbc:
                    raise KeyError(f"Retained mass fractions fret were given as a dictionary, but has no key 'fbc{r}'.")
                fret=self.get_fret_from_fbc(fbc[f"fbc{r}"],r)
                if np.isnan(fret[0]):
                    continue
                self.set_fret(m500c,fret,region=r,extrapolate500cto200m=extrapolate500cto200m)

    def set_fret_from_fb(self,m500c,fb,region,extrapolate500cto200m=False):
        """
        Transforms baryon fractions to retained mass fractions. Use either 
        this function, set_fret_from_fbc, or set_fret. This function has no 
        return value; retained mass fractions are set internally in the
        resummation object. Note that the retained mass fractions obtained are 
        linearly interpolated in log10(M500c) to the mass bins used for the 
        halo-matter cross spectra.

        Parameters
        ----------

        m500c: float or array_like
            Log10(M500c) halo masses for which baryon fractions are provided, 
            in units of Msun (no h).

        fb: type and length matching m500c
            The baryon fractions for the given halo masses within the radius 
            corresponding to the given region. Based on these the retained 
            mass fractions will be determined. The baryon fractions should be 
            given for a single region at a time, and be absolute mass 
            fractions, i.e. not normalized by Omega_b/Omega_m.

        region: str
            The corresponding overdensity region for the provided baryon 
            fractions.

        extrapolate500cto200m: bool, optional
            Whether baryon fractions given for R500c should be extrapolated 
            to estimate retained mass fractions within R200m as well, 
            according to the relation of the left-hand side of Figure 10 
            of VD25. If this parameter is set to True for regions other than 
            R500c it will be ignored.

        """
        if not self.initialized:
            raise RuntimeError("Resummation was initialized with errors; please create a new instance with compatible settings.")
        if isinstance(fb,dict):
            raise TypeError(f"Baryon fractions should not be given as a dictionary; this is only supported for corrected baryon fractions.")
        if not isinstance(fb,np.ndarray):
            fb=np.array(fb)
        if np.max(fb)>0.25:
            print(f"Warning: exceedingly high baryon fractions were given, please make sure these are unnormalized by Omega_b/Omega_m.")
        if np.min(fb)<0:
            raise ValueError(f"Baryon fractions cannot be negative.")
        fbc=(1-self.cosm["Ob"]/(self.cosm["Om"]-self.cosm["Onu"]))/(1-fb)
        self.set_fret_from_fbc(m500c,fbc,region=region,extrapolate500cto200m=extrapolate500cto200m)
    
    def set_fret(self,m500c,fret,region=None,extrapolate500cto200m=False,fret5x500_crit=None):
        """
        Sets the retained mass fractions directly. Use either this function, 
        set_fret_from_fb, or set_fret_from_fbc. This function has no return 
        value; retained mass fractions are set internally in the resummation 
        object. Note that the provided retained mass fractions are linearly 
        interpolated in log10(M500c) to the mass bins used for the halo-matter 
        cross spectra.

        Parameters
        ----------

        m500c: float or array_like
            Log10(M500c) halo masses for which baryon fractions are provided, 
            in units of Msun (no h).

        fret: type and length matching m500c
            The retained mass fractions for the given halo masses within 
            the radius corresponding to the given region. The retained mass 
            fractions can either be given for a single region, or as a 
            dictionary for all regions at once. If fbc is a dictionary, the 
            keys are expected to be of the form fbc200m, fbc500c, etc. In this 
            case, no region should be provided as a parameter.

        region: str, optional
            The corresponding overdensity region if fbc is a list or array of 
            floats. If fbc is given as a dict, this should be None.

        extrapolate500cto200m: bool, optional
            Whether baryon fractions given for R500c should be extrapolated 
            to estimate retained mass fractions within R200m as well, 
            according to the relation of the left-hand side of Figure 10 
            of VD25. If this parameter is set to True for regions other than 
            R500c it will be ignored.

        fret5x500_crit: float, optional
            The retained mass fraction within the combination of all 5xR500c 
            regions of all resolved haloes. This value should be redshift-
            dependent, but always close to unity. This value is completely 
            optional, and can also be set via set_fret5x500_crit(); see that 
            method for more information.

        """
        if not self.initialized:
            raise RuntimeError("Resummation was initialized with errors; please create a new instance with compatible settings.")
        if not isinstance(m500c,np.ndarray):
            m500c=np.array(m500c)
        if np.log10(np.min(m500c))>2:
            m500c=np.log10(m500c)
        if region is not None:
            if region not in self.regions:
                print(f"Warning: retained mass fractions given for region {region}, but this region was not set in initialization, and so the model will not use them.")
            if self.fret is None:
                self.fret={}
            self.fret.update({f"fret{region}": np.interp(self.m500c_cen,m500c,fret)})
            if region=="500_crit" and extrapolate500cto200m:
                if not "200_mean" in self.regions:
                    raise RuntimeError("To use extrapolate500cto200m please make sure to set 200_mean as one of the regions in initialization.")
                self.fret.update({"fret200_mean": np.interp(self.m500c_cen,m500c,self.extrapolate500cto200m(fret))})
        else:
            if not isinstance(fret,dict):
                raise TypeError("Retained mass fractions should be given in dictionary form, with region names as keys, or a region should be provided in the function call.")
            if self.fret is None:
                self.fret={}
            for r in self.regions:
                if not f"fret{r}" in fret:
                    raise KeyError(f"Retained mass fractions fret were given as a dictionary, but has no key 'fbc{r}'.")
                self.fret.update({f"fret{r}": np.interp(self.m500c_cen,m500c,fret[f"fret{r}"])})
            if extrapolate500cto200m:
                if not "500_crit" in self.regions or not "200_mean" in self.regions:
                    raise RuntimeError("To use extrapolate500cto200m please make sure to set 200_mean as one of the regions in initialization.")
                self.fret.update({"fret200_mean": np.interp(self.m500c_cen,m500c,self.extrapolate500cto200m(fret["fret500_crit"]))})
        if fret5x500_crit is not None:
            self.set_fret5x500_crit(fret5x500_crit)

    def set_fstar(self,m500c,fstar,region):
        """
        Sets stellar fractions within the given region. This is completely 
        optional, but allows for much more accurate results on scales 
        k>~10 h/Mpc. This function has no return value; stellar mass 
        fractions are set internally in the resummation object. Note that the 
        given stellar mass fractions are linearly interpolated in log10(M500c) 
        to the mass bins used for the halo-matter cross spectra. The stellar 
        fractions should be only set for a single region; calling this 
        function a second time overwrites stellar fractions set in the 
        previous call, whether the region is the same or not. To disable the 
        use of stellar fractions after they were already set, call this 
        again with fstar or region set to None. See section 2.2.3 of VD25 
        for more information on how stellar fractions are used in the model.

        Parameters
        ----------

        m500c: float or array_like
            Log10(M500c) halo masses for which stellar fractions are provided, 
            in units of Msun (no h).

        fstar: type and length matching m500c
            The stellar fractions for the given halo masses within the radius 
            corresponding to the given region. The stellar fractions should 
            be the stellar mass in the region divided by the total mass of 
            that region, and should only be set for a single region; 
            attempting to set the stellar fractions again overwrites both  
            these and the region associated with them. Set fstar to None to 
            disable the use of stellar fractions (equivalent to not calling 
            this function at all).

        region: str
            The corresponding overdensity region for the provided baryon 
            fractions. For the best results, this should be the inner-most 
            region set in initialization. Set region to None to disable the 
            use of stellar fractions (equivalent to not calling this function 
            at all).

        """
        if not self.initialized:
            raise RuntimeError("Resummation was initialized with errors; please create a new instance with compatible settings.")
        if not self.halospectra:
            print("Warning: stellar fractions will be ignored as no halo power spectra are available.")
            return
        if fstar is None or region is None:
            self.fstar=None
            self.stellarregion=None
            return
        if not isinstance(m500c,np.ndarray):
            m500c=np.array(m500c)
        if np.log10(np.min(m500c))>2:
            m500c=np.log10(m500c)
        if len(m500c)!=len(fstar):
            raise ValueError("Number of masses and stellar fractions given does not match.")
        if np.max(fstar)>1.0 or np.min(fstar)<0:
            raise ValueError("Stellar fractions must be relative to the total mass in the given regions, and should therefore be between zero and one.")
        if np.max(fstar)>0.16:
            print(f"Warning: exceedingly high stellar fractions were given, please make sure that this should be the stellar mass in the region divided by the total mass in the region.")
        if region not in self.regions:
            raise ValueError("Stellar fractions must be given inside a region the model was initialized for.")
        if region!=self.regions[-1]:
            print(f"Warning: stellar fractions should ideally be set for the innermost region.")
        self.fstar=np.interp(self.m500c_cen,m500c,fstar)
        self.stellarregion=region

    def set_fret5x500_crit(self,fret5x500_crit):
        """
        Sets the collective retained mass fraction for the combination of all 
        5xR500c halo regions. Setting this value is completely optional, see 
        section 3.6 of VD25. If this function is not called, or its parameter 
        is set to None, the 5xR500c power spectra are not used and the non-
        halo contribution is determined from the largest region set in 
        initialization instead.

        Parameters
        ----------

        fret5x500_crit: float
            The retained mass fraction within the combination of all 5xR500c 
            regions of all resolved haloes. This value should be redshift-
            dependent, but always close to unity.

        """
        if not self.initialized:
            raise RuntimeError("Resummation was initialized with errors; please create a new instance with compatible settings.")
        if "massfrac5x500_crittot" not in self.massfrac:
            print("Warning: retained mass fraction for the combined r5x500_crit region will be ignored as no mass fracion is available for it.")
            return
        if "p5x500_crittot" not in self.power:
            print("Warning: retained mass fraction for the combined r5x500_crit region will be ignored as no power spectrum is available for it.")
            return
        if isinstance(fret5x500_crit,list) or isinstance(fret5x500_crit,np.ndarray):
            raise TypeError("The large-scale retained mass fraction fret5x500_crit must be a scalar, if set at all.")
        self.fret5x500_crit=fret5x500_crit

    def run_resummation(self,k=None,raw=False): # Setting "raw" doesn't enforce convergence nor smooths the spectra
        """
        Calculates the model suppression signal at the given wavenumbers. 
        Baryon fractions should have been set before calling this function, 
        using either set_fret_from_fb(), set_fret_from_fbc(), or set_fret(). 
        Other settings, such as the cosmology to use, are set in the 
        initialization of the Resummation object.

        Parameters
        ----------

        k: array_like, optional
            The wavenumbers at which the resummation model should output the 
            suppression signal. These are assumed to be in units h/Mpc. If 
            none are provided, the wavenumbers at which cross spectra are 
            available are used.

        raw: bool, optional
            By default, the model enforces convergences at low k by setting 
            the suppression signal to unity for k<0.07 h/Mpc. It also smoothes 
            the resulting signal by convolving it with a Gaussian, with a 
            standard deviation equal to two data points. Setting raw to True 
            suppresses these steps.

        Returns
        -------

        kmodel: np.array
            The wavenumbers at which the suppression signal is returned, in 
            units h/Mpc. These are identical to the optional parameter k if it 
            was used. The array is also set as a property of the resummation 
            object.

        qqmodel: np.array
            The suppression signal as calculated by the resummation model for 
            the cosmology, redshift, and baryon fractions/retained mass 
            fractions set (dimensionless). The array is also set as a property 
            of the resummation object.

        """
        if not self.initialized:
            raise RuntimeError("Resummation was initialized with errors; please create a new instance with compatible settings.")
        if self.fret is None:
            raise RuntimeError("Retained mass fractions were not set, run set_fret(fret) or set_fb(fb) first.")
        for r in self.regions:
            if f"fret{r}" not in self.fret:
                raise RuntimeError("No retained mass fraction was set for region {r}.")
        self.ptotmodel=np.zeros(self.nk,dtype=np.float64)
        # Start with the contribution of the inner-most overdensity region
        for bin in np.arange(self.nbins):
            self.ptotmodel[:]+=self.fret[f"fret{self.regions[-1]}"][bin]*self.power[f"p{self.regions[-1]}"][bin,:]
        # Now add in any number of halo annuli
        for i in reversed(np.arange(len(self.regions)-1)):
            fretlarge=self.fret[f"fret{self.regions[i]}"]
            massfraclarge=self.massfrac[f"massfrac{self.regions[i]}"]
            biaslarge=self.bias[f"bias{self.regions[i]}"]
            powerlarge=self.power[f"p{self.regions[i]}"]
            fretsmall=self.fret[f"fret{self.regions[i+1]}"]
            massfracsmall=self.massfrac[f"massfrac{self.regions[i+1]}"]
            biassmall=self.bias[f"bias{self.regions[i+1]}"]
            powersmall=self.power[f"p{self.regions[i+1]}"]
            for bin in np.arange(self.nbins):
                self.ptotmodel[:]+=(fretlarge[bin]*massfraclarge[bin]*biaslarge[bin]-fretsmall[bin]*massfracsmall[bin]*biassmall[bin])/(massfraclarge[bin]*biaslarge[bin]-massfracsmall[bin]*biassmall[bin])*(powerlarge[bin,:]-powersmall[bin,:])
        # Optionally, include a stellar contribution
        if self.fstar is not None:
            for bin in np.arange(self.nbins):
                self.ptotmodel[:]+=self.fstar[bin]*self.fret[f"fret{self.stellarregion}"][bin]*(self.massfrac[f"massfrac{self.stellarregion}"][bin]/self.massfrac["massfrac500_crit"][bin]*self.power["p500haloes"][bin,:]-self.power[f"p{self.stellarregion}"][bin,:])
        # Next, add the non-halo contribution, i.e. matter outside the largest overdensity region, optionally split into <R5x500c and >R5x500c
        if self.fret5x500_crit is None:
            self.pnotmodel=(1-np.sum(self.fret[f"fret{self.regions[0]}"]*self.massfrac[f"massfrac{self.regions[0]}"]*self.bias[f"bias{self.regions[0]}"]))/(1-np.sum(self.massfrac[f"massfrac{self.regions[0]}"]*self.bias[f"bias{self.regions[0]}"]))*self.power["pouternot"]
        else:
            self.ptotmodel[:]+=(self.fret5x500_crit*self.massfrac["massfrac5x500_crittot"]*self.bias["bias5x500_crittot"]-np.sum(self.fret[f"fret{self.regions[0]}"]*self.massfrac[f"massfrac{self.regions[0]}"]))/(self.massfrac["massfrac5x500_crittot"]*self.bias["bias5x500_crittot"]-np.sum(self.massfrac[f"massfrac{self.regions[0]}"]*self.bias[f"bias{self.regions[0]}"]))*(self.power["p5x500_crittot"]-self.power["poutertot"])
            self.pnotmodel=(1-self.fret5x500_crit*self.massfrac["massfrac5x500_crittot"]*self.bias["bias5x500_crittot"])/(1-self.massfrac["massfrac5x500_crittot"]*self.bias["bias5x500_crittot"])*self.power["p5x500_critnot"]
        qmodel=(self.ptotmodel+self.pnotmodel)/self.power["ptot"]
        if not raw:
            qmodel[self.power["k"]<0.07]=1
        # Finally, square the resummed contributions
        self.qqmodel=qmodel**2
        self.ptotmodel*=qmodel
        self.pnotmodel*=qmodel
        # If a set of "k"-values was provided, interpolate the results to these (note: assumed to be in units of h/Mpc!)
        if k is not None:
            self.kmodel=k
            self.qqmodel=np.interp(np.log10(self.kmodel),np.log10(self.power["k"]),self.qqmodel)
        if not raw:
            if k is None:
                self.kmodel=np.logspace(-4,np.log10(np.max(self.power["k"])),num=500)
                self.qqmodel=np.interp(np.log10(self.kmodel),np.log10(self.power["k"]),self.qqmodel)
            self.qqmodel=gaussian_filter1d(self.qqmodel,2.0,mode='nearest')
        elif k is None:
            self.kmodel=self.power["k"]
        # Return scale factors k (units h/Mpc) and model baryonic-to-DMO power spectrum ratios (dimensionless)
        return self.kmodel,self.qqmodel
