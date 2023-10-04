# Reproduction-of-Multiple-Instance-Dictionary-Learning-for-Activity-Representation-study

Riproduzione del lavoro effettuato da *Sabanadesan Umakanthan, Simon Denman, Clinton Fookes and Sridha Sridharan, Multiple Instance Dictionary Learning for Activity
Representation*.

Si realizza un framework per la rappresentazione delle attività per "rinnovare" alcuni elementi del classico approccio Bag-Of-Words, considerati "migliorabili".
Per prima cosa si utilizza una variante ad istanza multipla di SVM (mi-SVM) per identificare feature positive per ogni categoria di attività e una classica implementazione k-means per generare un codebook.
Si applica poi una codifica lineare Locality-Constrained per la quantizzazione delle feature, seguita da uno spatio-temporal pyramid pooling per la conservazione di informazioni spazio-temporali.
Infine, si impiega un SVM per classificare i video.
