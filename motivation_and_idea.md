# Generació de Comic2Script

## Resum

Ser capaços d'agafar un cómic i fer la translació automaticament a guió.

## Com: 

0. Segmentació d'Escenes
1. Reconeixement d'Entitats (Visual): Quines entitats estan participant a cada escena*:
	a. Component de few shot retreieval; les entitats perduren al llarg d'escenes i edicions
	b. Reconeixement robust a canvis d'estil
2. Asociació de "speech": A quina entitat (visual) correspon cada seqüencia de text
3. Reconeixement de text
4. Reconeixement d'Entitats (textuals): De qui està parlant el texte*
5. Asociació d'entitats (textuals) a entitats (visuals): S'està parlant d'algú de l'escena?
6. Descripció d'escenaris: Que hi ha a l'escena (image captioning).

* Es pot estudiar la identitat d'una entitat (visual / textual) a partir de les sever relacions a través de _record linkage_?
Si tenim dues entitats amb relacions isomorfes a un altre entitat --> Son la mateixa entitat.

## Que:

Així amb totes aquestes tasques tenim el contexte de l'escena;
Els personatges que hi participen,
Les seves interaccions entre ells.

*Conclusió*: Tenim lo necesari per escriure un guió

## Tasques Implicades:

1. Layout Analysis (0., 1.)
2. Named Entity Recognition (1., 4., 5.)
3. Image Retrieval (1.)
4. Image Recogniton/Classification
5. Image-To-Text Recognition
6. IMage-To-Text Captioning
7. Object Detection (6., 1., 2., 0...)
8. Record Linkage

## Pasos Dependents:

(6), ((0, 1, 2) <--5--> (3, 4))

6 És independent
(0, 1, 2) Són fortament dependents
(3, 4) Són Fortament dependents
5 depèn de (0, 1, 2) i de (3, 4) però no a la inversa.
