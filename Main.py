import numpy as np
from matplotlib import pyplot as plt

# Funcțiile tale existente (nemodificate)
def citeste_grid_lungime(cale_fisier):
    with open(cale_fisier, 'r') as f:
        var = np.loadtxt(f, skiprows=1)
        np.set_printoptions(precision=16, suppress=True)
        return var

def citeste_grid_latime(cale_fisier1, cale_fisier2):
    c1 = open(cale_fisier1, 'r')
    c2 = open(cale_fisier2, 'r')
    c1 = np.loadtxt(c1, skiprows=1)
    c2 = np.loadtxt(c2, skiprows=1)
    np.set_printoptions(precision=16, suppress=True)
    vect = np.concatenate((c1, c2))
    return vect

def citeste(cale_fisier, puncte):
    with open(cale_fisier, 'rb') as f:
        grid_size = np.fromfile(f, dtype=np.int32, count=3)
        header = np.fromfile(f, dtype=np.int32, count=4)

        Nx = grid_size[0]
        Nr = grid_size[1]
        Nth = grid_size[2]

        size_per_var = Nx * Nr * Nth
        f.seek(size_per_var * 4, 1)
        floaturi = np.fromfile(f, dtype=np.float32, count=size_per_var)
        matrice = floaturi.reshape(Nx, Nr, Nth, order='F')
        matrice = matrice[:, :, 64]

    return grid_size, header, matrice

# Citirea fișierelor
cale_fisier1 = 'FLOW_phys_1_var_8_723500.raw'
cale_fisier2 = 'FLOW_phys_2_var_8_723500.raw'
grid_size1, header1, Mgrid1 = citeste(cale_fisier1, 73)
grid_size2, header2, Mgrid2 = citeste(cale_fisier2, 497)

Nx = grid_size1[0]
Nr1 = grid_size1[1]
Nr2 = grid_size2[1]
Nr = Nr1 + Nr2 # 570!
cale_fisier_grid1 = './FisierData2/z_grid_1.dat'
cale_fisier_grid2 = './FisierData2/r_grid_1.dat'
cale_fisier_grid3 = './FisierData2/r_grid_2.dat'

x = citeste_grid_lungime(cale_fisier_grid1)
r = citeste_grid_latime(cale_fisier_grid2, cale_fisier_grid3)
x = x[:Nx]
r = r[:Nr]

# Coordonatele radiale simetrice
r_aux = -r[::-1]  # -r
r_full = np.concatenate((r_aux, r))  # De la -r la +r

# Filtrăm r_full pentru a include doar valorile între -5 și 5
r_mask = (r_full >= -9) & (r_full <= 9)  # Creează o mască pentru intervalul dorit
r_filtered = r_full[r_mask]  # Filtrează r_full

# Filtrăm Mgrid_mirrored corespunzător
Mgrid_combined = np.concatenate((Mgrid1, Mgrid2), axis=1)  # (2581, 570)
Mgrid_mirrored = np.concatenate((Mgrid_combined[:, ::-1], Mgrid_combined), axis=1)  # (2581, 1140)
Mgrid_filtered = Mgrid_mirrored[:, r_mask]  # Filtrează datele pe axa r

# Creăm o nouă grilă 2D doar pentru intervalul filtrat
xx, rr = np.meshgrid(x, r_filtered)
xx = np.transpose(xx)
rr = np.transpose(rr)

# Plotare utilizând Matplotlib
plt.figure(figsize=(12, 5))
plt.pcolormesh(xx, rr, Mgrid_filtered, shading='auto', cmap="viridis")
plt.xlabel('x')
plt.ylabel('r')
plt.title('Reprezentare')
plt.colorbar(label='Viteză U')
plt.show()
print(Mgrid_combined.min())

