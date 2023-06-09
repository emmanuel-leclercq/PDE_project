import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class Mesh:
    def __init__(self, l):
        self.l = l
        self.vtx = None
        self.elt = None

    def GenerateRectangleMesh(self, Lx, Ly, Nx, Ny):
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

    def GenerateLShapeMesh(self, N):
        # Calculer le nombre de subdivisions pour chaque sous-domaine
        N1 = int(N * self.l)
        N2 = N - N1

        # Générer les maillages rectangulaires
        vtx1, elt1 = self.GenerateRectangleMesh(self.l, 1, N1, N)
        vtx2, elt2 = self.GenerateRectangleMesh(1.0-self.l, self.l, N2, N1)

        # Décaler le deuxième maillage en x
        vtx2[:, 0] += self.l

        # Combiner les deux maillages
        new_vtx = np.concatenate((vtx1, vtx2))
        # mettre à jour les indices des éléments du deuxième maillage
        elt2 += len(vtx1)
        new_elt = np.concatenate((elt1, elt2))

        self.vtx = new_vtx
        self.elt = new_elt

    def plot_mesh(self, val=None):
        plt.figure()

        if val is None:
            plt.triplot(self.vtx[:, 0], self.vtx[:, 1], self.elt)
        else:
            triang = mtri.Triangulation(
                self.vtx[:, 0], self.vtx[:, 1], self.elt)
            plt.tripcolor(triang, val, shading='flat', cmap='coolwarm')
            plt.colorbar(orientation='vertical')
            plt.triplot(triang, 'k--', alpha=0.3)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()


class PDE:
    def __init__(self, mesh, b=np.array([1, 1]), c=1):
        # Initialise le maillage, le vecteur b, la constante c (avec valeurs par défaut)
        # la matrice globale et le vecteur de terme source sont None,
        # il faut appeler les méthodes nécessaires pour les initialiser
        self.mesh = mesh
        self.b = b
        self.c = c
        self.global_matrix = None
        self.source_term = None
        self.solution = None

    def generate_rig_matrix(self):
        nbr_vtx = len(self.mesh.vtx)
        nbr_elt = len(self.mesh.elt)

        # Obtenir les coordonnées des sommets des triangles
        v0 = self.mesh.vtx[self.mesh.elt[:, 0]]
        v1 = self.mesh.vtx[self.mesh.elt[:, 1]]
        v2 = self.mesh.vtx[self.mesh.elt[:, 2]]

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
        I = np.repeat(self.mesh.elt[:, :, None], 3, axis=2)
        J = np.repeat(self.mesh.elt[:, None, :], 3, axis=1)
        data = B.flatten()

        # Créer la matrice COO à partir des données assemblées
        K = coo_matrix((data, (I.flatten(), J.flatten())),
                       shape=(nbr_vtx, nbr_vtx))

        return K

    def generate_second_matrix(self):
        # Initialize the matrix
        second_matrix = np.zeros(
            (self.mesh.vtx.shape[0], self.mesh.vtx.shape[0]))

        # Iterate over each element
        for el in self.mesh.elt:
            # Get the vertices of the element
            vertices = self.mesh.vtx[el]

            # Compute the area of the element
            area = 0.5 * \
                abs(np.cross(vertices[1] - vertices[0],
                    vertices[2] - vertices[0]))

            # Compute the gradient of the shape functions
            grad_N = np.array([[vertices[1, 1] - vertices[2, 1], vertices[2, 0] - vertices[1, 0]],
                               [vertices[2, 1] - vertices[0, 1],
                                   vertices[0, 0] - vertices[2, 0]],
                               [vertices[0, 1] - vertices[1, 1], vertices[1, 0] - vertices[0, 0]]]) / (2 * area)

            # Compute the local convection matrix
            local_second_matrix = area * (self.b[0] * np.outer(
                grad_N[:, 0], grad_N[:, 0]) + self.b[1] * np.outer(grad_N[:, 1], grad_N[:, 1]))

            # Add the local convection matrix to the global matrix
            for i in range(3):
                for j in range(3):
                    second_matrix[el[i], el[j]] += local_second_matrix[i, j]

        return second_matrix

    def generate_mass_matrix(self):
        # Nombre de sommets et d'éléments
        nbr_vtx = len(self.mesh.vtx)
        nbr_elt = len(self.mesh.elt)

        # Préparation des listes pour stocker les données de la matrice de masse
        rows = []
        cols = []
        data = []

        # Boucle sur les éléments
        for i in range(nbr_elt):
            # Indices des sommets de l'élément
            idx = self.mesh.elt[i]

            # Coordonnées des sommets de l'élément
            xa, ya = self.mesh.vtx[idx[0]]
            xb, yb = self.mesh.vtx[idx[1]]
            xc, yc = self.mesh.vtx[idx[2]]

            # Calcul de l'aire de l'élément (demi aire du parallélogramme défini par les sommets)
            area = 0.5 * abs((xa-xc)*(yb-ya) - (xa-xb)*(yc-ya))

            # Contribution de l'élément à la matrice de masse
            for j in range(3):
                for k in range(3):
                    if j == k:  # diagonal term
                        val = 2 * area / 12
                    else:  # off-diagonal term
                        val = area / 12

                    # Ajout à la liste des données de la matrice de masse
                    rows.append(idx[j])
                    cols.append(idx[k])
                    data.append(self.c * val)

        # Assemblage de la matrice de masse
        M = coo_matrix((data, (rows, cols)), shape=(nbr_vtx, nbr_vtx))

        return M

    def generate_global_matrix(self):
        self.global_matrix = csr_matrix(self.generate_rig_matrix(
        ) + self.generate_second_matrix() + self.generate_mass_matrix())

    def assemble_source_term(self, f):
        n = len(self.mesh.vtx)
        b = np.zeros(n)

        # Parcourir tous les éléments du maillage
        for i in range(len(self.mesh.elt)):
            element = self.mesh.elt[i]

            # Calculer la contribution de chaque élément
            for j in range(3):
                # Récupérer les sommets du triangle
                v1, v2, v3 = self.mesh.vtx[element]

                # Calculer l'aire du triangle
                area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                                 (v3[0] - v1[0]) * (v2[1] - v1[1]))

                # Calculer le centre du triangle
                centroid = (v1 + v2 + v3) / 3

                # Ajouter la contribution de l'élément à l'intégrale
                b[element[j]] += area * f(*centroid)

        self.source_term=b

    def solve(self, f):
        self.assemble_source_term(f)
        self.solution = spsolve(self.global_matrix, self.source_term)

    def plot_approximation(self, v_h):
        self.mesh.plot_mesh(val=v_h)

    def test_mass_matrix(self):
        # Générer la matrice de masse
        M = self.generate_mass_matrix()

        # Créer un vecteur u à partir de la fonction u(x) = x1 + x2 + beta
        u = np.ones(self.mesh.vtx.shape[0])

        # Calculer U^T * M * U
        Mu = M @ u
        UtMU = u.T @ Mu
        return f'la valeur UtMU est-elle égale de celle de l\'aire (2*l-l^2={2*self.mesh.l-self.mesh.l**2}): {abs(2*self.mesh.l-self.mesh.l**2-UtMU)==0}'

    def test_rig_matrix(self):
        # Générer la matrice de rigidité
        K = self.generate_rig_matrix()

        u = np.ones(self.mesh.vtx.shape[0])
        return f'K*u est-il proche du vecteur nul: {np.isclose(K * u, 0).all()}'

    def test_rig_matrix2(self, alpha, beta):
        # Générer la matrice de rigidité
        K = self.generate_rig_matrix()

        # Nombre de sommets
        nbr_vtx = len(self.mesh.vtx)

        # Créer les vecteurs U et V
        U = np.array([self.mesh.vtx[i, 0] + alpha for i in range(nbr_vtx)])
        V = np.array([self.mesh.vtx[i, 1] + beta for i in range(nbr_vtx)])

        # Calculer V^T * K * U
        result = V.T.dot(K.dot(U))

        return f'V^T * K * U = {result}, est-ce proche de 0: {np.isclose(result, 0)}'

    
    def plot_error(self, u_ex):
        # Calcul de la solution exacte sur les sommets du maillage
        x_nodes = self.mesh.vtx[:, 0]
        y_nodes = self.mesh.vtx[:, 1]

        # Effectuer une interpolation de Lagrange de u_ex aux noeuds de la grille
        u_ex_interpolated = np.interp(x=x_nodes,xp=x_nodes, fp=u_ex(x_nodes, y_nodes))
        # Calculer l'erreur u_h - Π_h(u_ex)
        error = self.solution - u_ex_interpolated
        # Plot the error
        self.plot_approximation(error)

    def compute_L2_norm(self, u):
        #calcul de la norme L2 
        areas = np.abs(np.cross(self.mesh.vtx[self.mesh.elt[:, 1], :] - self.mesh.vtx[self.mesh.elt[:, 0], :], self.mesh.vtx[self.mesh.elt[:, 2], :] - self.mesh.vtx[self.mesh.elt[:, 0], :])) / 2
        node_areas = np.zeros(len(self.mesh.vtx))
        np.add.at(node_areas, self.mesh.elt, areas[:, None] / 3)
        return np.sqrt(np.sum(node_areas * u**2))

    