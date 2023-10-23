#ifndef MESH_H
#define MESH_H

#include "vec3.h"
#include "hitinfo.h"
#include "ray.h"
#include "rtutility.h"

typedef struct UV{
    double u;
    double v;
} UV;

typedef struct Triangle{
    point3 A;
    point3 B;
    point3 C;
    material mat;
    UV uvA;
    UV uvB;
    UV uvC;
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

triangle* list_of_mesh(const char *filename, const char *mtl_file, int *numTriangles, int *numMaterials, char ***mat_path_list_ptr, int **quelSommetPourMat_list_ptr, int** quelMatPourSommet_list_ptr) {
    FILE *file = fopen(filename, "r");

    if (!file) {
        perror("\nImpossible d'ouvrir le fichier du mesh.\n");
        exit(EXIT_FAILURE);
    }

    char line[256];
    
    // compte le nombre de triangles, de materiaux et le nom du fichier mtl
    int nbTriangle = 0;
    int nbMaterial = 0;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'f') {
            nbTriangle++;
        }
        else if (line[0] == 'u' && line[1] == 's' && line[2] == 'e' && line[3] == 'm' && line[4] == 't' && line[5] == 'l') {
            nbMaterial++;
        }
    }
    rewind(file);
    
    // compte le nombre de sommets et de textures
    int nbSommets = 0;
    int nbTextures = 0;
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v') {
            if (line[1] == ' ') {
                nbSommets++;
            } else if (line[1] == 't') {
                nbTextures++;
            }
        }
    }
    rewind(file);

    *numTriangles = nbTriangle;
    *numMaterials = nbMaterial;
    triangle* mesh_list = (triangle*)malloc(nbTriangle * sizeof(triangle));
    point3* sommets_list = (point3*)malloc(nbSommets * sizeof(point3));
    UV* uv_list = (UV*)malloc(nbTextures * sizeof(UV));
    char** mat_path_list = (char**)malloc(nbMaterial * sizeof(char*));

    int* quelSommet_list = (int*)malloc(3 * nbTriangle * sizeof(int));
    int* quelTexture_list = (int*)malloc(3 * nbTriangle * sizeof(int));
    // premier sommet avec le material i
    *quelSommetPourMat_list_ptr = (int*)malloc(nbMaterial * sizeof(int));
    *quelMatPourSommet_list_ptr = (int*)malloc(nbTriangle * sizeof(int));

    int sommets_index = 0;
    int uv_index = 0;
    int quelSommet_ind = 0;
    int quelTextures_ind = 0;

    // lit les sommets et les textures
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v') {
            if (line[1] == ' ') {
                sscanf(line, "v %lf %lf %lf", &sommets_list[sommets_index].e[0], &sommets_list[sommets_index].e[1], &sommets_list[sommets_index].e[2]);
                sommets_index++;
            } else if (line[1] == 't') {
                sscanf(line, "vt %lf %lf", &uv_list[uv_index].u, &uv_list[uv_index].v);
                uv_index++;
            }
        }
    }
    rewind(file);

    // lis les faces et les matériaux
    int path_mat_ind = -1;
    char mat_name[256] = "";
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'f') {
            sscanf(line, "f %d/%d/%*d %d/%d/%*d %d/%d/%*d", &quelSommet_list[quelSommet_ind], &quelTexture_list[quelTextures_ind], &quelSommet_list[quelSommet_ind + 1], &quelTexture_list[quelTextures_ind + 1], &quelSommet_list[quelSommet_ind + 2], &quelTexture_list[quelTextures_ind + 2]);
            (*quelMatPourSommet_list_ptr)[quelSommet_ind/3] = path_mat_ind;
            quelSommet_ind += 3;
            quelTextures_ind += 3;
        }
        else if (line[0] == 'u' && line[1] == 's' && line[2] == 'e' && line[3] == 'm' && line[4] == 't' && line[5] == 'l') {
            sscanf(line, "usemtl %[^\n]", mat_name);
            mat_path_list[path_mat_ind+1] = tex_path_from_mtl(mtl_file, strdup(mat_name));
            (*quelSommetPourMat_list_ptr)[path_mat_ind+1] = quelSommet_ind/3;
            
            path_mat_ind++;
        }
    }

    for(int i = 0; i<nbMaterial; i++){
        printf("Triangle : [%d], Mat[%d] : %s\n", (*quelSommetPourMat_list_ptr)[i], i,  mat_path_list[i]);
    }

    // for(int i = 0; i<nbTriangle; i++){
    //     printf("Mat : [%d], Triangle : [%d] : %s\n", (*quelMatPourSommet_list_ptr)[i], i,  mat_path_list[(*quelMatPourSommet_list_ptr)[i]-1]);
    // }

    *mat_path_list_ptr = mat_path_list;
    // *quelSommetPourMat_list_ptr = quelSommetPourMaterial_list;

    int mesh_index = 0;
    for (int i = 0; i < nbTriangle; i++) {
        mesh_list[i].A = sommets_list[quelSommet_list[3 * i] - 1];
        mesh_list[i].B = sommets_list[quelSommet_list[3 * i + 1] - 1];
        mesh_list[i].C = sommets_list[quelSommet_list[3 * i + 2] - 1];
        mesh_list[i].uvA = uv_list[quelTexture_list[3 * i] - 1];
        mesh_list[i].uvB = uv_list[quelTexture_list[3 * i + 1] - 1];
        mesh_list[i].uvC = uv_list[quelTexture_list[3 * i + 2] - 1];

        mesh_list[i].mat = (material){SKY, BLACK, 0.0, 0.0};
    }

    printf("%s : %d triangles loaded\n", filename, nbTriangle);

    fclose(file);

    free(sommets_list);
    free(uv_list);
    free(quelSommet_list);
    free(quelTexture_list);
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