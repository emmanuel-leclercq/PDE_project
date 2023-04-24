import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import coo_matrix, diags

# 1.a


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
Lx, Ly, Nx, Ny = 0.7, 0.5, 30, 40

vtx, elt = GenerateRectangleMesh(Lx, Ly, Nx, Ny)
PlotMesh(vtx, elt)

# 1.b

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

# 1.c

def PlotMesh(vtx, elt, val=None):
    plt.figure()

    if val is None:
        plt.triplot(vtx[:, 0], vtx[:, 1], elt)
    else:
        cmap = plt.get_cmap('viridis')
        triang = mtri.Triangulation(vtx[:, 0], vtx[:, 1], elt)
        plt.tripcolor(triang, val, cmap=cmap,
                      shading='flat', edgecolors='k', lw=0.5)
        plt.colorbar()

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Maillage')
    plt.show()


# Exemple d'utilisation
N = 15
Nl = 6

vtx, elt = GenerateLShapeMesh(N, Nl)
val = np.zeros(vtx.shape[0])
PlotMesh(vtx, elt, val)

#2.b
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
    K = coo_matrix((data, (I.flatten(), J.flatten())),
                   shape=(nbr_vtx, nbr_vtx))

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

    C = coo_matrix((data, (I.flatten(), J.flatten())),
                   shape=(nbr_vtx, nbr_vtx))

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


def f_source(x, y, b_x, b_y, p, q, r):
    return np.exp((b_x * x + b_y * y) / 2) * np.sin(p * r * np.pi * x) * np.sin(q * r * np.pi * y)

#3.b
def assemble_rhs(vtx, elt, b_x, b_y, p, q, r):
    n_elts = elt.shape[0]
    n_nodes = vtx.shape[0]

    # Points et poids de Gauss
    gauss_points = np.array([[-np.sqrt(1/3), -np.sqrt(1/3)],
                             [np.sqrt(1/3), -np.sqrt(1/3)],
                             [np.sqrt(1/3),  np.sqrt(1/3)],
                             [-np.sqrt(1/3),  np.sqrt(1/3)]])
    gauss_weights = np.array([1, 1, 1, 1])

    # Initialisation du second membre (RHS)
    rhs = np.zeros(n_nodes)

    # Boucle sur les éléments
    for e in range(n_elts):
        nodes = elt[e, :]

        # Calcul de l'aire de l'élément
        v0, v1, v2 = vtx[nodes, :]
        area = 0.5 * \
            abs(np.linalg.det(
                np.array([[v1[0]-v0[0], v1[1]-v0[1]], [v2[0]-v0[0], v2[1]-v0[1]]])))

        # Boucle sur les points de Gauss
        for i, gp in enumerate(gauss_points):
            xi, eta = gp
            # Fonctions de forme
            N0 = 1 - xi - eta
            N1 = xi
            N2 = eta
            N = np.array([N0, N1, N2])

            # Coordonnées du point de Gauss dans l'élément
            x_gauss = np.dot(N, vtx[nodes, 0])
            y_gauss = np.dot(N, vtx[nodes, 1])

            # Évaluation de la fonction source f au point de Gauss
            f_gauss = f_source(x_gauss, y_gauss, b_x, b_y, p, q, r)

            # Mise à jour du second membre
            rhs[nodes] += area * f_gauss * N * gauss_weights[i]

    return rhs

#3.d
def PlotApproximation(vtx, elt, u_h, u_ex_proj):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Afficher la solution numérique u_h
    tri = mtri.Triangulation(vtx[:, 0], vtx[:, 1], triangles=elt)
    plot1 = ax[0].tripcolor(tri, u_h, shading='flat', cmap='viridis')
    fig.colorbar(plot1, ax=ax[0], orientation='vertical', label='u_h')
    ax[0].triplot(tri, 'k--', alpha=0.3)
    ax[0].set_title('Solution numérique u_h')

    # Afficher l'erreur u_h - Π_h u_ex
    error = np.abs(u_h - u_ex_proj)
    plot2 = ax[1].tripcolor(tri, error, shading='flat', cmap='viridis')
    fig.colorbar(plot2, ax=ax[1], orientation='vertical', label='Erreur')
    ax[1].triplot(tri, 'k--', alpha=0.3)
    ax[1].set_title('Erreur u_h - Π_h u_ex')

    plt.tight_layout()
    plt.show()


# Calculer les valeurs nodales de la solution exacte projetée
u_ex_proj = np.array([u_ex(v[0], v[1], b_x, b_y, p, q, r, alpha) for v in vtx])

# Appeler la fonction PlotApproximation
PlotApproximation(vtx, elt, u_numerical, u_ex_proj)


def L2_norm(vtx, elt, values):
    area = np.abs(
        np.cross(vtx[elt[:, 1]] - vtx[elt[:, 0]], vtx[elt[:, 2]] - vtx[elt[:, 0]])) / 2
    nodal_mean = np.mean(values[elt], axis=1)
    return np.sqrt(np.sum(area * nodal_mean**2))


refinements = np.arange(1, 11)  # Niveaux de raffinement du maillage
error_ratios = []

for refinement in refinements:
    # Générer un maillage avec le niveau de raffinement actuel
    vtx, elt = GenerateRectangleMesh(
        L_x, L_y, refinement * N_x, refinement * N_y)

    # Résoudre le problème pour le maillage actuel et obtenir la solution numérique u_h
    u_numerical = solve_problem(vtx, elt, b, c, p, q, r)

    # Calculer les valeurs nodales de la solution exacte projetée
    u_ex_proj = np.array(
        [u_ex(v[0], v[1], b_x, b_y, p, q, r, alpha) for v in vtx])

    # Calculer les normes L2 de l'erreur et de la solution exacte projetée
    error_norm = L2_norm(vtx, elt, u_numerical - u_ex_proj)
    u_ex_proj_norm = L2_norm(vtx, elt, u_ex_proj)

    # Calculer le rapport d'erreur et l'ajouter à la liste des rapports d'erreur
    error_ratios.append(error_norm / u_ex_proj_norm)

# Tracer la convergence de l'erreur
plt.figure()
plt.plot(refinements, error_ratios, 'o-', label="Rapport d'erreur")
plt.xlabel('Niveau de raffinement')
plt.ylabel('∥ u_h − Π_h u_ex ∥_L^2 (Ω) / ∥ Π_h u_ex∥L2(Ω)')
plt.title('Convergence de l\'erreur')
plt.legend()
plt.show()
