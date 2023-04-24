import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import coo_matrix, diags

def PlotMesh(vtx, elt, val=None):
    plt.figure()
    
    if val is None:
        plt.triplot(vtx[:, 0], vtx[:, 1], elt)
    else:
        cmap = plt.get_cmap('viridis')
        triang = tri.Triangulation(vtx[:, 0], vtx[:, 1], elt)
        plt.tripcolor(triang, val, cmap=cmap, shading='flat', edgecolors='k', lw=0.5)
        plt.colorbar()

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Maillage')
    plt.show()

def GenerateRectangleMesh(Lx, Ly, Nx, Ny):
    nbr_vtx = (Nx + 1) * (Ny + 1)
    nbr_elt = 2 * Nx * Ny

    vtx = np.zeros((nbr_vtx, 2))
    elt = np.zeros((nbr_elt, 3), dtype=int)

    dx = Lx / Nx
    dy = Ly / Ny

    # Générer les sommets
    vtx_index = 0
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            vtx[vtx_index] = [i * dx, j * dy]
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
            elt[elt_index] = [lower_left, upper_left, lower_right]
            elt_index += 1

            # Second triangle
            elt[elt_index] = [lower_right, upper_left, upper_right]
            elt_index += 1

    return vtx, elt

# Exemple d'utilisation
Lx = 0.7
Ly = 0.5
Nx = 30
Ny = 40

vtx, elt = GenerateRectangleMesh(Lx, Ly, Nx, Ny)
PlotMesh(vtx, elt)

def GenerateLShapeMesh(N, Nl):
    assert 0 < Nl <= N, "Nl doit être compris entre 0 et N."

    h = 1 / N
    l = Nl * h

    # Générer deux maillages rectangulaires
    vtx1, elt1 = GenerateRectangleMesh(l, 1, Nl, N)
    vtx2, elt2 = GenerateRectangleMesh(1, l, N, Nl)

    # Décaler les indices des éléments du deuxième maillage
    elt2 += len(vtx1)

    # Fusionner les sommets et les éléments des deux maillages
    vtx = np.vstack((vtx1, vtx2))
    elt = np.vstack((elt1, elt2))

    return vtx, elt

# Exemple d'utilisation
N = 15
Nl = 6

vtx, elt = GenerateLShapeMesh(N, Nl)
val=np.zeros(vtx.shape[0])
PlotMesh(vtx, elt, val)


def generate_rig_matrix(vtx, elt):
    nbr_vtx = len(vtx)
    nbr_elt = len(elt)

    # Obtenir les coordonnées des sommets des triangles
    v0 = vtx[elt[:, 0]]
    v1 = vtx[elt[:, 1]]
    v2 = vtx[elt[:, 2]]

    # Calculer les vecteurs des côtés des triangles et les aires des éléments
    e1 = v1 - v0
    e2 = v2 - v0
    areas = 0.5 * np.abs(np.cross(e1, e2))

    # Calculer les gradients locaux
    invA = 1 / (2 * areas)
    b = np.zeros((nbr_elt, 3, 2))
    b[:, 0] = (v2 - v1) * invA[:, None]
    b[:, 1] = (v0 - v2) * invA[:, None]
    b[:, 2] = (v1 - v0) * invA[:, None]

    B = np.einsum('ijk,ilk->ijl', b, b) * areas[:, None, None]

    # Assembler les matrices locales dans la matrice globale
    I = np.repeat(elt[:, :, None], 3, axis=2)
    J = np.repeat(elt[:, None, :], 3, axis=1)
    data = B.flatten()

    # Créer la matrice COO à partir des données assemblées
    K = coo_matrix((data, (I.flatten(), J.flatten())), shape=(nbr_vtx, nbr_vtx))

    return K


def generate_second_matrix(vtx, elt, b):
    nbr_vtx = len(vtx)
    nbr_elt = len(elt)

    # Calculer les matrices de gradients locaux pour chaque élément
    v0 = vtx[elt[:, 0]]
    v1 = vtx[elt[:, 1]]
    v2 = vtx[elt[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    areas = 0.5 * np.abs(np.cross(e1, e2))

    invA = 1 / (2 * areas)
    b_local = np.zeros((nbr_elt, 3, 2))
    b_local[:, 0] = (v2 - v1) * invA[:, None]
    b_local[:, 1] = (v0 - v2) * invA[:, None]
    b_local[:, 2] = (v1 - v0) * invA[:, None]

    C_local = np.einsum('ijk,i->ijk', b_local, b) * areas[:, None, None]

    # Assembler les matrices locales dans la matrice globale
    I = np.repeat(elt[:, :, None], 3, axis=2)
    J = np.repeat(elt[:, None, :], 3, axis=1)
    data = C_local.flatten()

    C = coo_matrix((data, (I.flatten(), J.flatten())), shape=(nbr_vtx, nbr_vtx))

    return C
    
    def generate_mass_matrix(vtx, elt, c):
    # Calculer les aires des éléments
    v0 = vtx[elt[:, 0]]
    v1 = vtx[elt[:, 1]]
    v2 = vtx[elt[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    areas = 0.5 * np.abs(np.cross(e1, e2))

    # Calculer les contributions locales de la matrice de réaction
    local_contrib = (areas * c / 3).repeat(3)

    # Assembler la matrice globale de réaction
    R = diags(local_contrib, shape=(len(vtx), len(vtx)))

    return R
    
    
