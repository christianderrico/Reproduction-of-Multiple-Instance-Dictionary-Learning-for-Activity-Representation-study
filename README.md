# Reproduction-of-Multiple-Instance-Dictionary-Learning-for-Activity-Representation-study

Riproduzione del lavoro effettuato da *Sabanadesan Umakanthan, Simon Denman, Clinton Fookes and Sridha Sridharan, Multiple Instance Dictionary Learning for Activity
Representation*.

Si realizza un framework per la rappresentazione delle attività per "rinnova" alcuni elementi del classico approccio Bag-Of-Words, considerati "migliorabili" dal team di ricerca.
Per prima cosa si utilizza una variante ad istanza multipla di SVM (mi-SVM) per identificare feature positive per ogni categoria di attività e poi una classica implementazione k-means per generare un codebook.
Si applica poi una codifica lineare Locality-Constrained per scrivere nei termini del dizionario generato le feature, seguita da uno spatio-temporal pyramid pooling per conservare informazioni spazio-temporali.
Infine si utilizza un SVM per classificare i video.
