from astropy import units as u
import numpy as np
#from read_lines_e import read_lines

#G = 4.302*10**-3        # pc, km/s, solar mass
#G = 1.90809*10**5        # solar radii, km/s, solar mass
# G = 887.3515302300001    # AU, km/s, solar mass
G = 1    # AU, km/s, solar mass
# G = 39.478           # AU3 * yr-2 * Msun-1
#M = 1200.0



def cart_2_kep(r_vec,v_vec, M):
    mu = G * M
    #1
    h_bar = np.cross(r_vec,v_vec)
    h = np.linalg.norm(h_bar)
    #2
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    #3
    E = 0.5*(v**2) - mu/r
    #4
    a = -mu/(2*E)
    #5
    e = np.sqrt(1 - (h**2)/(a*mu))
    #6
    i = np.arccos(h_bar[2]/h)
    #7
    omega_LAN = np.arctan2(h_bar[0],-h_bar[1])
    #8
    #beware of division by zero here
    lat = np.arctan2(np.divide(r_vec[2],(np.sin(i))),\
    (r_vec[0]*np.cos(omega_LAN) + r_vec[1]*np.sin(omega_LAN)))
    #9
    p = a*(1-e**2)
    nu = np.arctan2(np.sqrt(p/mu) * np.dot(r_vec,v_vec), p-r)
    #10
    omega_AP = lat - nu
    #11
    EA = 2*np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(nu/2))
    #12
    n = np.sqrt(mu/(a**3))

    t = 0

    T = t - (1/n)*(EA - e*np.sin(EA))

    return a,e,i,omega_AP,omega_LAN,T, EA

#if __name__ == "__main__":

#infile = open("state-vector13", "r") #reading text file to get data
#data = infile.readlines()
#infile.close()
#
#outfile = open("13-binary-elements-single.dat",'w')
#
#for line_variables in read_lines(data): #reading data from dwarf1_2.txt line by line in this iterative loop
#    for var_name, variable in line_variables.items():
#        globals()[var_name] = variable
##    print (x,y,z,vx,vy,vz)
#    x = float(x)
#    y = float(y)
#    z = float(z)
#    vx = float(vx)
#    vy = float(vy)
#    vz = float(vz)
#
#    r_test = np.array([x, y, z])
#    v_test = np.array([vx,vy,vz])
#    t = 0
#    output = cart_2_kep(r_test,v_test)
#    print(output)
#    semi,ecc,inc,omega_AP,omega_LAN,T,EA = output
#    inc = inc*(180.0/3.14159265359)
#    outfile.write(str(time)+" "+str(semi)+" "+str(ecc)+" "+str(inc)+'\n')
#    outfile.flush()
##    print(semi,ecc,inc)
