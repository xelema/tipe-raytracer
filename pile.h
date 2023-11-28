#ifndef PILE_H
#define PILE_H

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct IndicesRef{
   double n[2];
} IndRef;

typedef struct Element element;
struct Element {
    IndRef ind;
    element* suivant;
};

typedef struct Pile{
    element *premier;
} pile;

pile* init_pile(){
    pile* res = (pile*)malloc(sizeof(pile));
    res->premier = NULL;
    return res;
}

IndRef depiler(pile *pile)
{
    if (pile == NULL){
        exit(EXIT_FAILURE);
    }

    IndRef res;
    element *elementDepile = pile->premier;

    if (pile != NULL && pile->premier != NULL){
        res = elementDepile->ind;
        pile->premier = elementDepile->suivant;
        free(elementDepile);
    }

    return res;
}

void empiler(pile *pile, double n1, double n2){
    element* nv_elmt = (element*)malloc(sizeof(element));

    if (pile == NULL || nv_elmt == NULL){
        exit(EXIT_FAILURE);
    }
   
    nv_elmt->ind.n[0] = n1;
    nv_elmt->ind.n[1] = n2;

    nv_elmt->suivant = pile->premier;
    pile->premier = nv_elmt;
}

void index_suivant_pile(pile* pile, double n2){
   IndRef ancien_elmt = depiler(pile);
   double old_n2 = ancien_elmt.n[1];
   empiler(pile, ancien_elmt.n[0], ancien_elmt.n[1]);
   empiler(pile, old_n2, n2);
}

IndRef info_pile_actuelle(pile* pile){
    IndRef res = depiler(pile);
    empiler(pile, res.n[0], res.n[1]);
    return res;
}

#endif
