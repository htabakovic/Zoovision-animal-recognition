import cv2
import os

# ============================================================
# Objectif du programme :
# Reconnaissance d'animaux (chat, chien, cheval, girafe, mouton)
# en comparant une image test avec une base d'images,
# à l'aide de descripteurs ORB et d'un BFMatcher.
# ============================================================


# 1) Préparer image (lecture, resize, gris)
def preparer(chemin):
    # Lecture de l'image en couleur
    img = cv2.imread(chemin, cv2.IMREAD_COLOR)

    if img is None:
        return None, None

    # Redimensionnement pour avoir des images de la même taille = comparaison plus stable des descripteurs
    img = cv2.resize(img, (400, 300))

    # Conversion en niveaux de gris
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gris


# 2) ORB , permet:
# - de détecter des points d'intérêt
# - de calculer des descripteurs binaires robustes
orb = cv2.ORB_create()  


# 3) BFMatcher 
# BFMatcher compare les descripteurs entre deux images
# NORM_HAMMING est utilisé car ORB produit des descripteurs binaires
algoBF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  


# 4) Charger l'image de test

image_test = "TEST/test_chat02.jpg"   # <-- CHANGER ICI, POUR TESTER D'AUTRES IMAGES
imgT, grisT = preparer(image_test)

#Choix valides pour Tester depuis le fichier TEST:
# test_chat01.jpg
# test_chat02.jpg
# test_chien01.jpg
# test_cheval01.jpg
# test_girafe01.jpg

#Consulter le fichier TEST pour voir les autres images de test disponibles


# Détection des points clés et calcul des descripteurs de l'image test
ptsT, desT = orb.detectAndCompute(grisT, None)  


# 5) Parcourir la base et chercher la meilleure correspondance 
dossier_base = "BASE"

# Initialisation des variables pour stocker le meilleur résultat
meilleur_score = -1
meilleur_fichier = None
meilleurs_corresp = None
meilleur_imgB = None
meilleurs_ptsB = None

# Parcours de toutes les images de la base
for fichier in os.listdir(dossier_base):

    # On ignore les fichiers cachés
    if fichier.startswith("."):
        continue

    # Extraction du label depuis le nom du fichier (ex : chat_1.jpg = chat)
    label = fichier.split("_")[0].lower()

    # Filtrage : on ne garde que les classes autorisées
    if label not in ["chat", "chien", "cheval", "girafe", "mouton", "vache"]: #changeable si on veut rajouter/enlever des animaux dans la base de donées "BASE"
        continue

    # Chargement et préparation de l'image de la base
    chemin = os.path.join(dossier_base, fichier)
    imgB, grisB = preparer(chemin)

    # Détection des points clés et descripteurs de l'image base
    ptsB, desB = orb.detectAndCompute(grisB, None)

    # Sécurité, si aucun descripteur n'est détecté
    # alors score à 0 par exemple si l'image est flou
    if desT is None or desB is None:
        continue

    # knnMatch
    # Pour chaque descripteur de l'image test,
    # on cherche les k descripteurs les plus proches dans l'image base
    correspondances = algoBF.knnMatch(desT, desB, k=5)

    # Tri des correspondances par distance croissante
    # Plus la distance est faible, plus la correspondance est bonne
    matches_tri = sorted(correspondances, key=lambda x: x[0].distance)

    # Score simple :
    # On compte le nombre de bonnes correspondances
    # parmi les 25 meilleures
    top25 = matches_tri[:25]
    score = 0

    for m in top25:
        # On vérifie que la correspondance existe
        # et que la distance est inférieure à un seuil
        if len(m) > 0 and m[0].distance < 50: # seuil ajustable
            score += 1

    # Affichage du score pour chaque image de la base
    print(fichier, "=> score:", score)

    # Mise à jour du meilleur résultat
    if score > meilleur_score:
        meilleur_score = score
        meilleur_fichier = fichier
        meilleurs_corresp = matches_tri
        meilleur_imgB = imgB
        meilleurs_ptsB = ptsB


# 6) Résultat

# Si aucune image n'a donné de correspondance valable
if meilleur_fichier is None:
    print("Aucune correspondance trouvée.")
    exit()

# Prédiction finale basée sur le nom du fichier ayant le meilleur score
prediction = meilleur_fichier.split("_")[0].lower()

print("Prediction :", prediction)            # affiche la prédiction
print("Meilleure image :", meilleur_fichier) # affiche le nom de la meilleure image
print("Score :", meilleur_score)             # affiche le score de la meilleure correspondance


# Affichage des 25 meilleurs matches avec drawMatchesKnn qui permet de visualiser les points appariés
img_matches = cv2.drawMatchesKnn(imgT, ptsT, meilleur_imgB, meilleurs_ptsB,
                                 meilleurs_corresp[:25], None, flags=2)
cv2.imshow("Resultat", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
