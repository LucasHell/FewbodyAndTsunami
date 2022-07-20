import numpy as np
import sys
try:
    from tsunami import KeplerUtils
    NO_TSUNAMI = False
except ModuleNotFoundError:
    print("tsunami.py not found, skipping KeplerUtils orbit calculation")
    NO_TSUNAMI = True


def tsunami_calc(m, pv, i, j):
    KU = KeplerUtils()
    delta_pv = pv[i] - pv[j]
    mtot = m[i] + m[j]

    orb = KU.cart_to_kepl(delta_pv, mtot)
    print("BINARY:\na = {:g} au\ne = {:g}\ni = {:g}\nome = {:g}\nOme = {:g}\nnu = {:g}\n".format(*orb))

    pv_com = (pv[i]*m[i] + pv[j]*m[j])/mtot
    pv_out = pv[k]
    delta_pv_out = pv_com - pv_out
    mtot_out = m[i] + m[j] + m[k]

    horb = KU.cart_to_kepl(delta_pv_out, mtot_out)
    vinf = (- mtot_out / horb[0])**0.5
    print("HYPER:\na = {:g} au\ne = {:g}\ni = {:g}\nome = {:g}\nOme = {:g}\nnu = {:g}".format(*horb))
    print("vinf = {:g}".format(vinf))

def notsunami_calc(m, p, v, i, j):
    # Calculating specific energy of i-j pair
    print("Semi-major")

    dv = v[i] - v[j]
    mbin = m[i]+m[j]

    kinen = 0.5 * (dv*dv).sum()

    dp = p[i] - p[j]
    r = (dp*dp).sum()**0.5

    poten = - mbin / r

    energy = kinen + poten

    semi = - mbin / (2*energy)
    print("\nsemi = {:g}".format(semi))

    angmom = np.cross(dp, dv)
    angmom_mag = (angmom*angmom).sum()**0.5

    ecc = (1 + 2*angmom_mag**2 * energy / mbin**2)**0.5
    print("ecc = {:g}".format(ecc))

    ecc_vec = np.cross(dv, angmom) / mbin - dp/r
    print("ecc, alt = {:g}".format((ecc_vec*ecc_vec).sum()**0.5))


lids = [0, 1, 2]

fname = sys.argv[1]
i, j = 1, 2
if len(sys.argv) > 2:
    i = int(sys.argv[2])
    j = int(sys.argv[3])
    lids.remove(i)
    lids.remove(j)
k = lids[0]
coords = np.loadtxt(fname)

m = coords[:,6]
p = coords[:,0:3]
v = coords[:,3:6]

pv = coords[:,0:6]

if not NO_TSUNAMI:
    tsunami_calc(m, pv, i, j)

notsunami_calc(m, p, v, i, j)

