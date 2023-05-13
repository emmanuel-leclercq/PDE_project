import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def GenerateRectangleMesh(self,Lx, Ly, Nx, Ny):
    nbr_vtx = (Nx + 1) * (Ny + 1)
    nbr_elt = 2 * Nx * Ny

    new_vtx = np.zeros((nbr_vtx, 2))
    new_elt = np.zeros((nbr_elt, 3), dtype=int)

    dx = Lx / Nx
    dy = Ly / Ny

    # Générer les sommets
    vtx_index = 0
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            new_vtx[vtx_index] = [i * dx, j * dy]
            vtx_index += 1

    # Générer les éléments
    elt_index = 0
    for j in range(Ny):
        for i in range(Nx):
            # Indices des sommets du rectangle courant
            lower_left = j * (Nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + (Nx + 1)
            upper_right = upper_left + 1

            # Premier triangle
            new_elt[elt_index] = [lower_left, upper_left, lower_right]
            elt_index += 1

            # Second triangle
            new_elt[elt_index] = [lower_right, upper_left, upper_right]
            elt_index += 1

    return new_vtx, new_elt

def GenerateLShapeMesh(self,N):
    # Calculer le nombre de subdivisions pour chaque sous-domaine
    N1 = int(N * self.l)
    N2 = N - N1

    # Générer les maillages rectangulaires
    vtx1, elt1 = GenerateRectangleMesh(self.l,1,N1, N)
    vtx2, elt2 = GenerateRectangleMesh(1.0-self.l,self.l,N2, N1)

    # Décaler le deuxième maillage en x
    vtx2[:, 0] += self.l

    # Combiner les deux maillages
    new_vtx = np.concatenate((vtx1, vtx2))
    elt2 += len(vtx1)  # mettre à jour les indices des éléments du deuxième maillage
    new_elt = np.concatenate((elt1, elt2))

    self.vtx = new_vtx
    self.elt = new_elt