#ifndef TEXTURE_H
#define TEXTURE_H

#include <string.h>
#include "hitinfo.h"
#include "mesh.h"

color checker_texture_value(color c1, color c2, double scale, point3 p) {
    int x = (int)floor(1.0 / scale * p.e[0]);
    int y = (int)floor(1.0 / scale * p.e[1]);
    int z = (int)floor(1.0 / scale * p.e[2]);
    if ((x + y + z) % 2 == 0) return c1;
    else return c2;
}

point3 get_barycentric_coord(triangle tri, HitInfo hitInfo){
    point3 res;
    double areaABC = vec3_dot(hitInfo.normal, vec3_cross(sub(tri.B, tri.A), sub(tri.C, tri.A)));
    double areaPBC = vec3_dot(hitInfo.normal, vec3_cross(sub(tri.B, hitInfo.hitPoint), sub(tri.C, hitInfo.hitPoint)));
    double areaPCA = vec3_dot(hitInfo.normal, vec3_cross(sub(tri.C, hitInfo.hitPoint), sub(tri.A, hitInfo.hitPoint)));

    res.e[0] = areaPBC / areaABC;
    res.e[1] = areaPCA / areaABC;
    res.e[2] = 1.0 - res.e[0] - res.e[1];

    return res;
}

color get_texture_color(triangle tri, HitInfo hitInfo, material* mat_list, int texture_width, int texture_height) {
    UV uv;
    point3 bary = get_barycentric_coord(tri, hitInfo);

    // coordonnée uv du point (entre 0 et 1)
    uv.u = (bary.e[0]*tri.uvA.u + bary.e[1]*tri.uvB.u + bary.e[2]*tri.uvC.u);
    uv.v = (bary.e[0]*tri.uvA.v + bary.e[1]*tri.uvB.v + bary.e[2]*tri.uvC.v);

    // coordonnées correspondant à la texture
    int x = (int)(uv.u * (double)(texture_width));
    int y = (int)(uv.v * (double)(texture_height));

    return mat_list[y * texture_width + x].diffuseColor;
}

material tri_uvmapping(triangle tri, HitInfo hitInfo, material* mat_list, int texture_width, int texture_height, int indice_tri, int* quelMatPourSommet) {
    UV uv;
    point3 bary = get_barycentric_coord(tri, hitInfo);

    uv.u = (bary.e[0] * tri.uvA.u + bary.e[1] * tri.uvB.u + bary.e[2] * tri.uvC.u);
    uv.v = (bary.e[0] * tri.uvA.v + bary.e[1] * tri.uvB.v + bary.e[2] * tri.uvC.v);

    // // Gère les répétitions de textures (uv > 1 ou uv < 0)
    uv.u = fmod(uv.u, 1.0); // fmod renvoie le reste, donc ici la partie décimale
    uv.v = fmod(uv.v, 1.0);
    if (uv.u < 0) {
        uv.u += 1.0;
    }
    if (uv.v < 0) {
        uv.v += 1.0;
    }

    int x = (int)(uv.u * (double)(texture_width));
    int y = (int)(uv.v * (double)(texture_height));

    int index = (y * texture_width + x) + (texture_height * texture_width * quelMatPourSommet[indice_tri]);
    // printf("quelMatPourSommet[%d] : %d\n", indice_tri , quelMatPourSommet[indice_tri]);
    material res;
    res = mat_list[index];

    // if (quelMatPourSommet[indice_tri] == 2){
    //     res.emissionStrength =  5;
    //     res.emissionColor = mat_list[index].diffuseColor;
    // }

    // if (quelMatPourSommet[indice_tri] == 7){
    //     res.emissionStrength =  3.5;
    //     res.emissionColor = mat_list[index].diffuseColor;
    // }

    // if (quelMatPourSommet[indice_tri] == 16){
    //     res.reflectionStrength = 0.30;
    // }
    

    // if (quelMatPourSommet[indice_tri] == 2 || quelMatPourSommet[indice_tri] == 12 || quelMatPourSommet[indice_tri] == 25){
    //     res.emissionColor = mat_list[index].diffuseColor;
    //     res.emissionStrength = 32.0;
    //     res.reflectionStrength = 0.0;
    // }

    // if (quelMatPourSommet[indice_tri] == 42){
    //     res.emissionColor = mat_list[index].diffuseColor;
    //     res.emissionStrength = 24.0;
    //     res.reflectionStrength = 0.0;
    // }

    // if (quelMatPourSommet[indice_tri] == 27){
    //     res.emissionColor = mat_list[index].diffuseColor;
    //     res.emissionStrength = 8.0;
    //     res.reflectionStrength = 0.0;
    // }

    // if (quelMatPourSommet[indice_tri] == 50 || quelMatPourSommet[indice_tri] == 53){
    //     if ((res.diffuseColor.e[0]*255+res.diffuseColor.e[1]*255+res.diffuseColor.e[2]*255) > 288){
    //         res.emissionColor = mat_list[index].diffuseColor;
    //         res.emissionStrength = 48.0;
    //         res.reflectionStrength = 0.0;
    //     }
    // }

    // if (quelMatPourSommet[indice_tri] == 5) {
    //     res.emissionColor = mat_list[index].diffuseColor;
    //     res.reflectionStrength = 0.0;
    //     res.emissionStrength = 5.0;
    // }

    return res;
}

material sphere_uvmapping(sphere s, HitInfo hitInfo, material* mat_list, int sky_width, int sky_height) {
    
    UV uv;
    vec3 d = divide(sub(hitInfo.hitPoint, s.center), s.radius);
    
    double theta = acos(-d.e[1]);
    double phi = atan2(-d.e[2], d.e[0]) + PI;

    uv.u = phi/(2*PI);
    uv.v = theta/PI;

    int x = (int)(uv.u * (double)(sky_width));
    int y = (int)(uv.v * (double)(sky_height));

    int index = (y * sky_width + x);
    // printf("quelMatPourSommet[%d] : %d\n", indice_tri , quelMatPourSommet[indice_tri]);
    material res;
    res = mat_list[index];

    return res;
}

color* create_tex_list(const char *filename, int *width, int *height) {
    FILE *file = fopen(filename, "r");

    char format[3];
    fscanf(file, "%s", format);

    fscanf(file, "%d %d", width, height);
    int maxVal;
    fscanf(file, "%d", &maxVal);

    color* tex_list = (color *)malloc((*width)*(*height)*sizeof(color));  

    int index = 0;
    for (int i = (*height)-1; i >= 0; i--){
        for (int j = 0; j < (*width); j++){
            index = i * (*width) + j;
            tex_list[index].e[0] = 0;
            tex_list[index].e[1] = 0;
            tex_list[index].e[2] = 0;
            fscanf(file, "%lf %lf %lf", &(tex_list)[index].e[0], &tex_list[index].e[1], &tex_list[index].e[2]);

            tex_list[index].e[0] /= maxVal;
            tex_list[index].e[1] /= maxVal;
            tex_list[index].e[2] /= maxVal;
        }
    }
    fclose(file);
    return tex_list;
}


material* create_mat_list(const char *filename, int *width, int *height) {
    FILE *file = fopen(filename, "r");

    char format[3];
    fscanf(file, "%s", format);

    fscanf(file, "%d %d", width, height);
    int maxVal;
    fscanf(file, "%d", &maxVal);

    material* tex_list = (material *)malloc((*width)*(*height)*sizeof(material));  

    int index = 0;
    for (int i = (*height)-1; i >= 0; i--){
        for (int j = 0; j < (*width); j++){
            index = i * (*width) + j;
            tex_list[index].diffuseColor.e[0] = 0;
            tex_list[index].diffuseColor.e[1] = 0;
            tex_list[index].diffuseColor.e[2] = 0;
            fscanf(file, "%lf %lf %lf", &(tex_list)[index].diffuseColor.e[0], &tex_list[index].diffuseColor.e[1], &tex_list[index].diffuseColor.e[2]);

            tex_list[index].diffuseColor.e[0] /= maxVal;
            tex_list[index].diffuseColor.e[1] /= maxVal;
            tex_list[index].diffuseColor.e[2] /= maxVal;
        }
    }
    fclose(file);
    return tex_list;
}

material* create_mat_list_mtl(char **path_file, int *width, int *height, int nbMaterials) {
    
    int global_ind_mult = 0;

    char* input = path_file[0];
    const char* extension = ".png";
    size_t extension_len = strlen(extension);
    const char* extension_pos = strstr(input, extension);

    size_t input_len = extension_pos - input;

    char* texfile_path = (char*)malloc(input_len + strlen(".ppm") + 1);
    char* alphafile_path = (char*)malloc(input_len + strlen("_alpha.ppm") + 1);

    strncpy(texfile_path, input, input_len);
    texfile_path[input_len] = '\0';

    strncpy(alphafile_path, input, input_len);
    alphafile_path[input_len] = '\0';

    strcat(texfile_path, ".ppm");
    strcat(alphafile_path, "_alpha.ppm");

    // printf("Nom du chemin tex[%d] : %s\n", 0, texfile_path);
    // printf("Nom du chemin alpha[%d] : %s\n", 0, alphafile_path);

    FILE *tex_file = fopen(texfile_path, "r");
    FILE *alpha_file = fopen(alphafile_path, "r");

    if (tex_file == NULL || alpha_file == NULL){
        free(texfile_path);
        free(alphafile_path);
        return NULL;
    }

    char format[3];
    fscanf(tex_file, "%s", format);
    fscanf(alpha_file, "%s", format);

    fscanf(tex_file, "%d %d", width, height);
    fscanf(alpha_file, "%d %d", width, height);

    int maxVal, max_alphaVal;
    fscanf(tex_file, "%d", &maxVal);
    fscanf(alpha_file, "%d", &max_alphaVal);

    material* res_mat_list = (material*)malloc((*width) * (*height) * nbMaterials * sizeof(material));
    material *mat_list = (material *)malloc((*width) * (*height) * sizeof(material));

    int index = 0;

    // diffuse mat
    for (int i = (*height)-1; i >= 0 ; i--){
        for (int j = 0; j < (*width); j++){
            index = i * (*width) + j;
            mat_list[index].diffuseColor.e[0] = 0;
            mat_list[index].diffuseColor.e[1] = 0;
            mat_list[index].diffuseColor.e[2] = 0;
            fscanf(tex_file, "%lf %lf %lf", &(mat_list)[index].diffuseColor.e[0], &mat_list[index].diffuseColor.e[1], &mat_list[index].diffuseColor.e[2]);
            fscanf(alpha_file, "%lf %*lf %*lf", &(mat_list)[index].alpha);

            mat_list[index].diffuseColor.e[0] /= maxVal;
            mat_list[index].diffuseColor.e[1] /= maxVal;
            mat_list[index].diffuseColor.e[2] /= maxVal;

            mat_list[index].alpha /= max_alphaVal;
            // printf("mat_list[%d].alpha = %lf\n", index, mat_list[index].alpha);

            int global_ind = index+((*height)*(*width)*global_ind_mult);
            res_mat_list[global_ind].diffuseColor = mat_list[index].diffuseColor;
            res_mat_list[global_ind].alpha = mat_list[index].alpha;
            res_mat_list[global_ind].emissionStrength = 0.0;

            // // si lumiere
            // res_mat_list[global_ind].emissionColor = mat_list[index].diffuseColor;
            // res_mat_list[global_ind].emissionStrength = 5.0;
        }
    }

    fclose(tex_file);
    fclose(alpha_file);
    free(texfile_path);
    free(alphafile_path);
    free(mat_list);
    global_ind_mult++;

    //premier tour de boucle manuel pour initialiser res_mat_list avec la tex_width et tex_height
    if (nbMaterials>1){
        for (int k = 1; k < nbMaterials; k++) {
            char* input = path_file[k];
            const char* extension = ".png";
            size_t extension_len = strlen(extension);
            const char* extension_pos = strstr(input, extension);

            if (extension_pos == NULL) {
                free(res_mat_list);
                return NULL;
            }

            size_t input_len = extension_pos - input;

            char* texfile_path = (char*)malloc(input_len + strlen(".ppm") + 1);
            char* alphafile_path = (char*)malloc(input_len + strlen("_alpha.ppm") + 1);

            strncpy(texfile_path, input, input_len);
            texfile_path[input_len] = '\0';

            strncpy(alphafile_path, input, input_len);
            alphafile_path[input_len] = '\0';

            strcat(texfile_path, ".ppm");
            strcat(alphafile_path, "_alpha.ppm");

            // printf("Nom du chemin tex[%d] : %s\n", k, texfile_path);
            // printf("Nom du chemin alpha[%d] : %s\n", k, alphafile_path);
            
            FILE* tex_file = fopen(texfile_path, "r");
            FILE* alpha_file = fopen(alphafile_path, "r");

            if (tex_file == NULL || alpha_file == NULL) {
                free(res_mat_list);
                free(texfile_path);
                free(alphafile_path);
                return NULL;
            }

            char format[3];
            fscanf(tex_file, "%s", format);
            fscanf(alpha_file, "%s", format);

            fscanf(tex_file, "%d %d", width, height);
            fscanf(alpha_file, "%d %d", width, height);

            int maxVal, max_alphaVal;
            fscanf(tex_file, "%d", &maxVal);
            fscanf(alpha_file, "%d", &max_alphaVal);

            material* mat_list = (material*)malloc((*width) * (*height) * sizeof(material));
            
            int index = 0;

            // diffuse mat
            for (int i = (*height)-1; i >= 0 ; i--){
                for (int j = 0; j < (*width); j++){
                    index = i * (*width) + j;
                    mat_list[index].diffuseColor.e[0] = 0;
                    mat_list[index].diffuseColor.e[1] = 0;
                    mat_list[index].diffuseColor.e[2] = 0;
                    fscanf(tex_file, "%lf %lf %lf", &(mat_list)[index].diffuseColor.e[0], &mat_list[index].diffuseColor.e[1], &mat_list[index].diffuseColor.e[2]);
                    fscanf(alpha_file, "%lf %*lf %*lf", &(mat_list)[index].alpha);

                    mat_list[index].diffuseColor.e[0] /= maxVal;
                    mat_list[index].diffuseColor.e[1] /= maxVal;
                    mat_list[index].diffuseColor.e[2] /= maxVal;
                    // printf("mat_list[%d].diffuse = %lf %lf %lf\n", index, mat_list[index].diffuseColor.e[0], mat_list[index].diffuseColor.e[1], mat_list[index].diffuseColor.e[2]);

                    mat_list[index].alpha /= max_alphaVal;
                    // printf("mat_list[%d].alpha = %lf\n", index, mat_list[index].alpha);

                    int global_ind = index+((*height)*(*width)*global_ind_mult);
                    res_mat_list[global_ind].diffuseColor = mat_list[index].diffuseColor;
                    res_mat_list[global_ind].alpha = mat_list[index].alpha;
                    res_mat_list[global_ind].emissionStrength = 0.0;

                    // // si lumiere
                    // res_mat_list[global_ind].emissionColor = mat_list[index].diffuseColor;
                    // res_mat_list[global_ind].emissionStrength = 5.0;
                }
            }

            fclose(tex_file);
            fclose(alpha_file);
            free(texfile_path);
            free(alphafile_path);
            free(mat_list);
            global_ind_mult++;
        }
    }
    return res_mat_list;
}

#endif