#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "vec3.h"
#include "hitinfo.h"
#include "ray.h"

typedef struct Triangle{
    point3 A;
    point3 B;
    point3 C;
    material mat;
} triangle;

// __host__ __device__ HitInfo hit_triangle(triangle tri, ray r){
// // https://youtu.be/Fr328siHuVc pour explications

//     HitInfo hitInfo;
//     hitInfo.didHit=false; 

//     vec3 normal = vec3_normalize(vec3_cross(sub(tri.B, tri.A), sub(tri.C, tri.A)));
//     double t = vec3_dot(sub(tri.C, r.origin), normal) / vec3_dot(r.dir, normal);

//     if (t >= 0.000001){
//         hitInfo.hitPoint = add(r.origin, multiply_scalar(r.dir, hitInfo.dst));

//         vec3 u = sub(tri.B, tri.A);
//         vec3 v = sub(tri.C, tri.A);
//         vec3 w = sub(hitInfo.hitPoint, tri.A);

//         double m11 = vec3_length_squared(u);
//         double m12 = vec3_dot(u, v);
//         double m22 = vec3_length_squared(v);
//         double detm = m11*m22 - m12*m12;

//         double b11 = vec3_dot(w, u);
//         double b21 = vec3_dot(w, v);
//         double detb = b11*m22 - b21*m12;

//         double beta = detb / detm; // coordonnée barycentrique par rapport à B
        
//         double g12 = b11;
//         double g22 = b21;
//         double detg = m11*g22 - m12*g12;

//         double gamma = detg / detm; // coordonnée barycentrique par rapport à C

//         double alpha = 1 - beta - gamma; // coordonnée barycentrique par rapport à A

//         if (alpha < 0 || alpha > 1) return hitInfo;
//         if (beta < 0 || beta > 1) return hitInfo;
//         if (gamma < 0 || gamma > 1) return hitInfo;

//         hitInfo.didHit = true;
//         hitInfo.normal = normal;
//         hitInfo.dst = t;
//     }
//     return hitInfo;
// }

HitInfo hit_triangle(triangle tri, ray r){

    point3 edgeAB = sub(tri.B, tri.A);
    point3 edgeAC = sub(tri.C, tri.A);
    vec3 normalVect = vec3_cross(edgeAB, edgeAC);

    vec3 ao = sub(r.origin, tri.A);  
    vec3 dao = vec3_cross(ao, r.dir);

    double det = -vec3_dot(r.dir, normalVect);
    double invDet = 1/det;

    double dst = vec3_dot(ao, normalVect) * invDet;
    double u = vec3_dot(edgeAC, dao) * invDet;
    double v = -vec3_dot(edgeAB, dao) * invDet;
    double w = 1 - u - v;

    HitInfo hitInfo;
    hitInfo.didHit = det >= 1E-6 && dst >= 0.00001 && u >= 0.00001 && v >= 0.00001 && w >= 0.00001;
    hitInfo.hitPoint = add(r.origin, multiply_scalar(r.dir, dst));
    hitInfo.normal = vec3_normalize(normalVect);
    hitInfo.dst = dst;

    return hitInfo;

}
triangle* list_of_mesh(const char *filename, int *numTriangles) {
    FILE *file = fopen(filename, "r");

    char line[256];
    int triangleCount = 0;

    // compte le nombre de triangles
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'f') {
            triangleCount++;
        }
    }
    rewind(file);
    
    // compte le nombre de sommets
    int nbSommets = 0;
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            nbSommets++;
        }
    }
    rewind(file);


    *numTriangles = triangleCount;
    triangle* mesh_list = (triangle*)malloc(triangleCount * sizeof(triangle));
    point3* sommets_list = (point3*)malloc(nbSommets*sizeof(point3));
    int* quelSommet_list = (int*)malloc(3*triangleCount*sizeof(int));
    
    int sommets_index = 0;    

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            sscanf(line, "v %lf %lf %lf", &sommets_list[sommets_index].e[0], &sommets_list[sommets_index].e[1], &sommets_list[sommets_index].e[2]);
            sommets_index++;
        }
    }

    int quelSommet_ind = 0;
    printf("\n");

    rewind(file);
    // while (fgets(line, sizeof(line), file)) {
    //     if (line[0] == 'f') {
    //         sscanf(line, "f %d/%*d/%*d %d/%*d/%*d %d/%*d/%*d", &quelSommet_list[quelSommet_ind], &quelSommet_list[quelSommet_ind+1], &quelSommet_list[quelSommet_ind+2]);
    //         quelSommet_ind += 3;
    //     }
    // }

        while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'f') {
            sscanf(line, "f %d//%*d %d//%*d %d//%*d", &quelSommet_list[quelSommet_ind], &quelSommet_list[quelSommet_ind+1], &quelSommet_list[quelSommet_ind+2]);
            quelSommet_ind += 3;
        }
    }

    int mesh_index = 0;
    for(int i = 0; i<triangleCount; i++){
        mesh_list[i].A = sommets_list[quelSommet_list[3*i]-1];
        mesh_list[i].B = sommets_list[quelSommet_list[3*i+1]-1];
        mesh_list[i].C = sommets_list[quelSommet_list[3*i+2]-1];
        mesh_list[i].mat = (material){SKY, BLACK, 0.0, 0.5};
    }

    printf(filename);
    printf(" : %d triangles loaded\n\n", triangleCount);

    // printf("nb sommets : %d \n", nbSommets);
    // printf("nb triangle : %d \n", triangleCount);
    // printf("\n");
    // for(int i = 0; i<nbSommets; i++){
    //     printf("sommets_list[%d] : %lf %lf %lf\n", i, sommets_list[i].e[0], sommets_list[i].e[1], sommets_list[i].e[2]);
    // }
    // printf("\n");
    // for(int i = 0; i<3*triangleCount; i++){
    //     printf("quelSommet_list[%d] : %d\n", i, quelSommet_list[i]);
    // }
    // printf("\n");
    // for(int i = 0; i<triangleCount; i++){
    //     printf("mesh_list[%d] : (%lf, %lf, %lf) (%lf, %lf, %lf) (%lf, %lf, %lf)\n", i, mesh_list[i].A.e[0], mesh_list[i].A.e[1], mesh_list[i].A.e[2],mesh_list[i].B.e[0], mesh_list[i].B.e[1], mesh_list[i].B.e[2], mesh_list[i].C.e[0], mesh_list[i].C.e[1], mesh_list[i].C.e[2]);
    // }
    // printf("\n");

    fclose(file);
    return mesh_list;
}

void move_mesh(double x, double y, double z, triangle** mesh_list, int nbTriangle){
    for (int i = 0; i<nbTriangle; i++){
        (*mesh_list)[i].A.e[0] += x;
        (*mesh_list)[i].B.e[0] += x;
        (*mesh_list)[i].C.e[0] += x;

        (*mesh_list)[i].A.e[1] += y;
        (*mesh_list)[i].B.e[1] += y;
        (*mesh_list)[i].C.e[1] += y;

        (*mesh_list)[i].A.e[2] += z;
        (*mesh_list)[i].B.e[2] += z;
        (*mesh_list)[i].C.e[2] += z;
    }
}


#endif