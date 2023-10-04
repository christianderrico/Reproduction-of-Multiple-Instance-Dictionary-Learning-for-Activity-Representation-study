# Reproduction-of-Multiple-Instance-Dictionary-Learning-for-Activity-Representation-study

Riproduzione del lavoro effettuato da *S. Umakanthan, S. Denman, C. Fookes and S. Sridharan, "Multiple Instance Dictionary Learning for Activity
Representation"*.

Si realizza un framework per il riconoscimento delle attività nei video, allo scopo di "rinnovare" alcuni elementi del classico approccio Bag-Of-Words, considerati "migliorabili".
L'approccio è strutturato così:
- si utilizza una variante ad istanza multipla di SVM (mi-SVM) per identificare feature positive per ogni categoria di attività e una classica implementazione k-means per la generazione dei codebook (uno per ogni classe).
- si applica una codifica lineare Locality-Constrained per la quantizzazione delle feature, seguita da uno spatio-temporal pyramid pooling per la conservazione di informazioni spazio-temporali.
- Infine, si impiega un SVM per effettuare la classificazione dei video.
