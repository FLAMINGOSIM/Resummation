import numpy as np

def getflamingolegend(simname,box="L1000N1800"):
    names=np.array(['HYDRO_FIDUCIAL_L1000N1800','HYDRO_FIDUCIAL_L2800N5040','HYDRO_FIDUCIAL_L1000N0900','HYDRO_FIDUCIAL_L1000N3600','HYDRO_WEAK_AGN_L1000N1800','HYDRO_STRONG_AGN_L1000N1800','HYDRO_STRONGER_AGN_L1000N1800','HYDRO_STRONGEST_AGN_L1000N1800','HYDRO_STRONG_SUPERNOVA_L1000N1800','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA_L1000N1800','HYDRO_JETS_L1000N1800','HYDRO_STRONG_JETS_L1000N1800','HYDRO_PLANCK_L1000N1800','HYDRO_PLANCK_LARGE_NU_FIXED_L1000N1800','HYDRO_PLANCK_LARGE_NU_VARY_L1000N1800','HYDRO_LOW_SIGMA8_L1000N1800','HYDRO_JETS_non_fixed_dir_L1000N1800','HYDRO_STRONG_JETS_non_fixed_dir_L1000N1800','HYDRO_JETS_published_L1000N1800','HYDRO_STRONG_JETS_published_L1000N1800','HYDRO_ADIABATIC_L1000N1800','HYDRO_LOW_SIGMA8_STRONGEST_AGN_L1000N1800','HYDRO_PLANCK_LARGER_NU_FIXED_L1000N1800'])
    cols=np.array(['#117733','#332288','#DDCC77','#CC6677','#abd0e6','#6aaed6','#3787c0','#105ba4','#FF8C40','#CC4314','#cfff4b','#55e1ce','#44AA99','#999933','#AA4499', '#882255','#7EFF4B','#55E18E','#7EFF4B','#55E18E','#000000','#7B68EE','#FF0000'])
    papernames=np.array(["L1$\\_$m9","L2p8$\\_$m9","L1$\\_$m10","L1$\\_$m8","fgas$+2\\sigma$","fgas$-2\\sigma$","fgas$-4\\sigma$","fgas$-8\\sigma$","M*$-\\sigma$","M*$-\\sigma$_fgas$-4\\sigma$","Jet","Jet_fgas$-4\\sigma$","Planck","PlanckNu0.24Fix","PlanckNu0.24Var","LS8","Jet","Jet_fgas$-4\\sigma$","Jet","Jet_fgas$-4\\sigma$","No cooling","LS8_fgas$-8\\sigma$","PlanckNu0.48Fix"])
    name=simname+"_"+box
    mask=(names==name)
    i=np.where(mask)[0]
    if len(i)==0:
        raise ValueError("FLAMINGO simulation named "+name+" not found.")
    i=i[0]
    return (papernames[i],cols[i])

def getflamingocosmology(simname):
    if "PLANCK_LARGER_NU_FIXED" in simname:
        return {"h": 0.673, "Om":0.316, "OL":0.684, "Ob":0.0494, "Mnu":0.48, "Onu":11.4e-3, "sigma8":0.709, "S8":0.728}
    elif "PLANCK_LARGE_NU_FIXED" in simname:
        return {"h": 0.673, "Om":0.316, "OL":0.684, "Ob":0.0494, "Mnu":0.24, "Onu":5.69e-3, "sigma8":0.769, "S8":0.789}
    elif "PLANCK_LARGE_NU_VARY" in simname:
        return {"h": 0.662, "Om":0.328, "OL":0.672, "Ob":0.0510, "Mnu":0.24, "Onu":5.87e-3, "sigma8":0.772, "S8":0.807}
    elif "PLANCK_MID_NU_VARY" in simname:
        return {"h": 0.673, "Om":0.316, "OL":0.684, "Ob":0.0496, "Mnu":0.12, "Onu":2.85e-3, "sigma8":0.800, "S8":0.821}
    elif "PLANCK_DCDM12" in simname:
        return {"h": 0.673, "Om":0.274, "OL":0.726, "Ob":0.0494, "Mnu":0.06, "Onu":1.42e-3, "sigma8":0.794, "S8":0.759}
    elif "PLANCK_DCDM24" in simname:
        return {"h": 0.673, "Om":0.239, "OL":0.726, "Ob":0.0494, "Mnu":0.06, "Onu":1.42e-3, "sigma8":0.777, "S8":0.694}
    elif "PLANCK" in simname:
        return {"h": 0.673, "Om":0.316, "OL":0.684, "Ob":0.0494, "Mnu":0.06, "Onu":1.42e-3, "sigma8":0.812, "S8":0.833}
    elif "LS8" in simname or "LOW_SIGMA8" in simname:
        return {"h": 0.682, "Om":0.305, "OL":0.695, "Ob":0.0473, "Mnu":0.06, "Onu":1.39e-3, "sigma8":0.760, "S8":0.766}
    elif "WMAP9" in simname:
        return {"h": 0.700, "Om":0.2793, "OL":0.7207, "Ob":0.0463, "Mnu":0.0, "Onu":0.0, "sigma8":0.821, "S8":0.792}
    elif "FIDUCIAL" in simname or "DESY3" in simname or "HYDRO_" in simname:
        return {"h": 0.681, "Om":0.306, "OL":0.694, "Ob":0.0486, "Mnu":0.06, "Onu":1.39e-3, "sigma8":0.807, "S8":0.815}
    else:
        raise ValueError(f"Cosmology {simname} is not recognized.")
