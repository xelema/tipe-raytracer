#ifndef DENOISER_H
#define DENOISER_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <OpenImageDenoise/oidn.h>

#include "vec3.h"
#include "ray.h"
#include "hitinfo.h"
#include "sphere.h"
#include "rtutility.h"
#include "camera.h"

HitInfo closest_hit(ray r, sphere* spheres, int nbSpheres);

void render_normal_albedo(vec3* normal, color* albedo, int hauteur_image, int largeur_image, camera cam, sphere* spheres, int nbSpheres){
    for (int j = hauteur_image - 1; j >= 0; j--){
        for (int i = 0; i < largeur_image; i++){
            int pixel_index = j*largeur_image+i;

            double u = (double)i/(largeur_image-1);
            double v = (double)j/(hauteur_image-1);

            ray r = get_ray(u, v, cam);
            HitInfo hitInfo = closest_hit(r, spheres, nbSpheres);

            if (hitInfo.didHit){
                normal[pixel_index] = hitInfo.normal;

                if (hitInfo.mat.emissionStrength > 0.0){
                    color HSL = rgb_to_hsl(hitInfo.mat.emissionColor);
                    HSL.e[2] *= 1.20; // luminosité
                    HSL.e[1] *= 1.20; // saturation (valeurs subjectives)
                    color newCol = hsl_to_rgb(HSL);
                    albedo[pixel_index] = newCol;
                }
                else{
                    albedo[pixel_index] = hitInfo.mat.diffuseColor;
                }
            }
            else{
                normal[pixel_index] = (color)BLACK;
                albedo[pixel_index] = (color)BLACK;
            }
        }
    }
}

void denoiser(int largeur_image, int hauteur_image, color* canva, camera cam, sphere* spheres, int nbSpheres){

    // tableau pour avoir chaque valeur de pixel au bon endroit, avant rebond de lumière (necessaire au denoise)
    color* albedo = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        albedo[i] = (color)BLACK;
    }

    // tableau pour avoir la normale de chaque objet (necessaire au denoise)
    color* normal = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        normal[i] = (color)BLACK;
    }

    render_normal_albedo(normal, albedo, hauteur_image, largeur_image, cam, spheres, nbSpheres);

    // initialise le device OIDN
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
    oidnCommitDevice(device);

    OIDNBuffer colorBuf  = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));
    OIDNBuffer albedoBuf = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));
    OIDNBuffer normalBuf = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));

    float* floatImage = (float*)oidnGetBufferData(colorBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatImage[3*i] = (float)canva[i].e[0] / 255.0f;
        floatImage[3*i+1] = (float)canva[i].e[1] / 255.0f;
        floatImage[3*i+2] = (float)canva[i].e[2] / 255.0f;
    }

    float* floatAlbedo = (float*)oidnGetBufferData(albedoBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatAlbedo[3*i] = (float)albedo[i].e[0];
        floatAlbedo[3*i+1] = (float)albedo[i].e[1];
        floatAlbedo[3*i+2] = (float)albedo[i].e[2];
    }

    float* floatNormal = (float*)oidnGetBufferData(normalBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatNormal[3*i] = (float)normal[i].e[0];
        floatNormal[3*i+1] = (float)normal[i].e[1];
        floatNormal[3*i+2] = (float)normal[i].e[2];
    }

    // crée le filtre ("RT" est pour raytracing)
    OIDNFilter filter = oidnNewFilter(device, "RT");
    oidnSetFilterImage(filter, "color",  colorBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // base
    oidnSetFilterImage(filter, "albedo", albedoBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // albedo (couleure sans lumiere)
    oidnSetFilterImage(filter, "normal", normalBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // normal
    oidnSetFilterImage(filter, "output", colorBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // denoise
    oidnCommitFilter(filter);

    // applique le filtre
    oidnExecuteFilter(filter);

    // check les erreurs du denoise
    const char* errorMessage;
    if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE){
        printf("Error: %s\n", errorMessage);
    }

    // retransfère les valeurs dans canva
    for (int i = 0; i < largeur_image * hauteur_image; i++) {
        canva[i].e[0] = (int)(floatImage[3*i] * 255.0f);
        canva[i].e[1] = (int)(floatImage[3*i+1] * 255.0f);
        canva[i].e[2] = (int)(floatImage[3*i+2] * 255.0f);
    }

    oidnReleaseBuffer(colorBuf);
    oidnReleaseBuffer(albedoBuf);
    oidnReleaseBuffer(normalBuf);
    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);
}

#endif