#ifndef RTUTILITY_H
#define RTUTILITY_H

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "ray.h"
#include "vec3.h"

// constantes

const double ratio = 16.0/9.0;

const int largeur_image = 1000;
const int hauteur_image = (int)(largeur_image / ratio);



const int nbRayonParPixel = 100;
const int nbRebondMax = 2;

const int profondeurMax = 5;

const color red = {{1, 0, 0}};
const color green = {{0, 1, 0}};
const color blue = {{0, 0, 1}};
const color white = {{1, 1, 1}};
const color black = {{0, 0, 0}};
const color sky = {{0.784, 0.965, 1}};

const double pi = 3.1415926535897932385;

// fonctions

double randomDouble1(){
    return (rand()/(RAND_MAX + 1.0)); // retourne un double random entre 0 inclu et 1 exclu
}

double randomDouble(double min, double max){
    return min + (max-min)*randomDouble1();
}



#endif
