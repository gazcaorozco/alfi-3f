from matplotlib import pyplot as plt
import numpy as np

epss = {}
vel_errors = {}
k_nrefs = [(2,1),(2,2),(2,3),(3,1)]#,(3,2)]#(2,4)

for params in k_nrefs:
    try:
        fname0 =  'output/vel_error_k_%i_nref_%i.out'%(params[0],params[1])
        epss[params],vel_errors[params]  = np.transpose(np.loadtxt(fname0, unpack=True))
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
l0, = plt.loglog(epss[(2,2)], vel_errors[(2,2)], marker = "v", markersize = 10)
l1, = plt.loglog(epss[(2,1)], vel_errors[(2,1)], marker = "o")
l2, = plt.loglog(epss[(2,3)], vel_errors[(2,3)], marker = "x")
#l3, = plt.loglog(epss[(2,4)], vel_errors[(2,4)], marker = "s")
l4, = plt.loglog(epss[(3,1)], vel_errors[(3,1)], marker = "*")
#l5, = plt.loglog(epss[(3,2)], vel_errors[(3,2)], marker = "D")

#plt.xlim(0, 1)
#plt.ylim(0.000001, 0.05)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'$\varepsilon$',fontsize=23)
plt.ylabel(r'$\|\mathbf{u}-\mathbf{u}_{e}\|_{L^2(\Omega)}$', fontsize=23)
plt.grid(True)
#plt.title('Velocity '+r'$L^2$'+' Error')
#plt.legend( (l0,l1,l2), (r'$Bn = 0.0$',r'$Bn = 2.0$',r'$Bn = 4.0$'), loc='upper right', shadow=True,prop={'size': 15})
plt.legend( (l1,l0,l2,l4), (r'$k=2,l =1$',r'$k=2,l =2$',r'$k=2,l =3$',r'$k=3,l =1$'), loc='lower right', shadow=True,prop={'size': 18})
aq.tight_layout()
plt.savefig('output/bingham_poiseuille_vel_errors.png')
plt.savefig('output/bingham_poiseuille_vel_errors.eps')

