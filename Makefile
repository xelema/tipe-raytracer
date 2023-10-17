CC = gcc
CFLAGS = -O3
LIBS = -lm -lpthread -lOpenImageDenoise
SRC = main.c
HEADERS = vec3.h ray.h hitinfo.h sphere.h rtutility.h camera.h denoiser.h mesh.h texture.h

prog: $(SRC) $(HEADERS)
	$(CC) -o $@ $(SRC) $(CFLAGS) $(LIBS)

clean:
	rm -f prog