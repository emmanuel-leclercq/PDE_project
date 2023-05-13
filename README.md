# PDE_project

On a définit deux classes de structures simples pour résoudre le problème

La classe Mesh, qui contient:
-un attribut l (le paramètre dans la définition de omega)
-un attribut vtx (matrice de coordonnées) valant None par défaut
-un attribut elt (matrice de connectivité) valant None par défaut

-Une méthode GenerateRectangleMesh(self,Lx, Ly, Nx, Ny) qui génère un maillage rectangulaire
avec en paramètre la longueur horizontale, longueur verticale, nombre de subdivision horizontale, nombre de subdivision verticales, et renvoie vtx, elt
-Une méthode GenerateLShapeMesh(self,N) se servant de la méthode précédente et prenant en paramètre le nombre de subdivision voulu, et qui initialise les attributs de classe vtx et elt
-Une méthode plot_mesh(self, val=None) permettant la visualisation du maillage

la classe PDE, qui contient:
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

On définit également deux fonctions en dehors des classes:
-Une fonction plot_approximation(v_h,vtx,elt) qui permet l'affichage d'un champ linéaire par morceau v_h sur un maillage, on l'utilise dans le fichier main.py pour représenter la solution numérique u_h
-Une fonction plot_convergence()