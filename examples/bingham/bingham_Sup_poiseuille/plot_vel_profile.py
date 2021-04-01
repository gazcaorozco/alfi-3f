from matplotlib import pyplot as plt
import numpy as np

params = (2,3)

epss = {}
vel_errors = {}
vel_profiles = {}

try:
    fname0 =  'output/vel_error_k_%i_nref_%i.out'%(params[0],params[1])
    epss[params],vel_errors[params]  = np.transpose(np.loadtxt(fname0, unpack=True))
except:
    err_msg = "Could not load file %s." % fname0 \
            + " Did you forget to run the code?"
    raise Exception(err_msg)

for eps_ in epss[params]:
    try:
        fname0 =  'output/vel_profile_k_%i_nref_%i_eps_%f.out'%(params[0],params[1],eps_)
        y_slice,vel_profiles[eps_]  = np.transpose(np.loadtxt(fname0, unpack=True))
    except:
        err_msg = "Could not load file %s." % fname0 \
                + " Did you forget to run the code?"
        raise Exception(err_msg)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', titlesize=20)
aq = plt.figure(1)                  
plt.clf()                      # clear figure
#lines = {}
#markers = ["o","v","x","s","*","D"]
#m = 0
l0, = plt.plot(y_slice, vel_profiles[epss[params][0]],linestyle = (0, (3, 1, 1, 1)))#, marker = "o")
l1, = plt.plot(y_slice, vel_profiles[epss[params][1]],linestyle = '-.')#, marker = "v")
l2, = plt.plot(y_slice, vel_profiles[epss[params][2]],linestyle = '--')#, marker = "x")
l3, = plt.plot(y_slice, vel_profiles[epss[params][3]],linestyle = (0, (1,1)))#, marker = "s")
#l4, = plt.plot(y_slice, vel_profiles[epss[params][4]],linestyle = 'dotted')#, marker = "*")
l5, = plt.plot(y_slice, vel_profiles[epss[params][5]])#, marker = "D")

#Exact solution
vel_exact = np.piecewise(y_slice, [y_slice <= -0.5, y_slice >= 0.5], [lambda y_slice: -y_slice**2 - y_slice, lambda y_slice: -y_slice**2 + y_slice, 0.25])
lex, = plt.plot(y_slice, vel_exact)

plt.xlim(-1, 1)
#plt.ylim(0.000001, 0.05)
plt.xlabel(r'$x_2$', fontsize=23)
plt.ylabel(r'$u_1(x_2)$', fontsize=23)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
#plt.title('Velocity '+r'$L^2$'+' Error')
#plt.legend( (l0,l1,l2), (r'$Bn = 0.0$',r'$Bn = 2.0$',r'$Bn = 4.0$'), loc='upper right', shadow=True,prop={'size': 15})
plt.legend( (l0,l1,l2,l3,l5,lex), (r'$\varepsilon = %f$'%epss[params][0],r'$\varepsilon = %f$'%epss[params][1],r'$\varepsilon= %f$'%epss[params][2],r'$\varepsilon= %f$'%epss[params][3],r'$\varepsilon= %f$'%epss[params][5],r'$\varepsilon = 0$'), loc='lower center', shadow=True,prop={'size': 17})
aq.tight_layout()
plt.savefig('output/bingham_poiseuille_vel_profiles.png')
plt.savefig('output/bingham_poiseuille_vel_profiles.eps')

