# Projet d'approximation numérique de la solution d'une équation différentielle par la méthode des éléments finis

On a définit deux classes de structures simples dans le fichier mesh_and_PDE_classes.py pour résoudre le problème


## La classe Mesh, qui contient:

-un attribut l (le paramètre dans la définition de omega)

-un attribut vtx (matrice de coordonnées) valant None par défaut

-un attribut elt (matrice de connectivité) valant None par défaut


-Une méthode GenerateRectangleMesh(self,Lx, Ly, Nx, Ny) qui génère un maillage rectangulaire
avec en paramètre la longueur horizontale, longueur verticale, nombre de subdivision horizontale, nombre de subdivision verticales, et renvoie vtx, elt

-Une méthode GenerateLShapeMesh(self,N) se servant de la méthode précédente et prenant en paramètre le nombre de subdivision voulu, et qui initialise les attributs de classe vtx et elt

-Une méthode plot_mesh(self, val=None) permettant la visualisation du maillage



## la classe PDE, qui contient:

-L'attribut mesh, objet de la classe précédemment décrite

-le vecteur b de l'équation, avec pour valeur par défaut (1,1)

-la constante c>0 avec pour valeur par défaut 1

-l'attribut global_matrix valant None par défaut, qui correspond au A dans la résolution AU=F

-L'attribut source_term valant None par défaut, qui correspond au terme source F dans AU=F

-L'attribut solution valant None par défaut, qui correspond au vecteur solution numérique de l'équation


-Une méthode generate_rig_matrix() qui renvoie la matrice de rigidité (1e terme de la forme sesquilinéaire)

-Une méthode generate_second_matrix() qui renvoie la matrice du second terme de la forme sesquilinéaire

-Une méthode generate_mass_matrix(): qui renvoie la matrice de masse

-Une méthode generate_global_matrix() qui initialise l'attribut global_matrix

-Une méthode assemble_source_term(f) qui initialise l'atttribut source_term pour la fonction source f

-Une méthode solve(f) qui résoud numériquement l'équation lorsque la fonction source vaut f, cela retourne un vecteur

-Une méthode test_mass_matrix() qui teste si UtMU est proche de l'aire (=l)

-Une méthode test_rig_matrix() qui teste si le vecteur KU est proche du vecteur nul

-Une méthode test_rig_matrix(alpha,beta) qui teste si VtKU est proche de zéro

-Une méthode plot_approximation(v_h) qui permet l'affichage d'un champ linéaire par morceau v_h sur le maillage


On définit également la fonction plot_convergence(rafinements=np.array([10,25,50,100])) qui trace la convergence de l'erreur avec les niveaux de rafinements pris en paramètre sous forme d'un vecteur, en dehors des classes dans le fichier main.ipynb (fichier notebook qui permet de simplifier l'affichage des graphiques).


Utilisation:

-Créer un objet my_mesh=Mesh(l=...)

-Utiliser la méthode my_mesh.GenerateLShapeMesh(N=...) 

-Créer un objet my_pde=PDE(Mesh=my_mesh)

-Utiliser la méthode my_pde.generate_global_matrix()

-Utiliser la méthode my_pde.solve(f) où f est la fonction source souhaitée, la solution est stockée dans my_pde.solution
