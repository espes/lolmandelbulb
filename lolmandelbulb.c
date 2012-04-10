// Niel van der Westhuizen, 2012

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define TRUE 1
#define FALSE 0

typedef struct _bmpHeader {
    uint32_t fileSize;
    uint16_t creator1;
    uint16_t creator2;
    uint32_t dataOffset;
} __attribute__((packed)) bmpHeader;

typedef struct _bmpInfoHeader {
    uint32_t headerSize; //40
    
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    
    //0 for RGB
    uint32_t compression;
    
    //can be left 0 for RGB
    uint32_t imageSize;
    
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    
    //for indexed colours
    uint32_t userColours;
    uint32_t importantColours;
} __attribute__((packed)) bmpInfoHeader;

typedef struct _colour {
    uint8_t b, g, r;
} colour;

typedef struct _vec2 {
    double x, y;
} vec2;

typedef struct _vec3 {
    double x, y, z;
} vec3;


typedef struct _raymarchResult {
    int hit;
    double distance;
    vec3 pos;
} raymarchResult;


double vecLength(vec3 v);
void vecNormalizeInplace(vec3 *v);
void vecRotateXInplace(vec3 *v, double theta);
void vecRotateYInplace(vec3 *v, double theta);
vec3 vecAddX(vec3 v, double c);
vec3 vecAddY(vec3 v, double c);
vec3 vecAddZ(vec3 v, double c);


void renderMandelbrot(FILE *outFile, int width, int height,
                      int zoom, double x, double y);
colour drawMandelbrot(double x, double y);
colour drawMandelbulb(double x, double y);

raymarchResult mandelbulbRaymarch(vec3 pos, vec3 direction);
double mandelbulbDistanceEstimator(vec3 pos);
vec3 mandelbulbEstimateNormal(vec3 pos);

void writeBMP(FILE *outFile, colour* data, int width, int height);

int main(int argc, char **argv) {
    FILE *outFile;
    int width = 512;
    int height = 512;
    
    int zoom;
    double x, y;
    
    if (argc < 5) {
        printf("Usage: %s outfile x y zoom [width] [height]\n", argv[0]);
        return 1;
    }
    
    //process the arguments
    outFile = fopen(argv[1], "wb");
    x = strtod(argv[2], NULL);
    y = strtod(argv[3], NULL);
    zoom = atoi(argv[4]);
    
    //optional arguments
    if (argc >= 6) {
        width = atoi(argv[5]);
    }
    if (argc >= 7) {
        height = atoi(argv[6]);
    }
    
    renderMandelbrot(outFile, width, height, zoom, x, y);
    
    return 0;
}

void renderMandelbrot(FILE *outFile, int width, int height,
                      int zoom, double x, double y) {

    double cx, cy;
    int px, py;
    double pixSize = 1. / (1 << zoom);
    colour data[height][width];
    
#pragma omp parallel for shared(data) private(px, py, cx, cy)
    for (py=0; py<height; py++) {
        for (px=0; px<width; px++) {
            cx = x + (px - width/2) * pixSize;
            cy = y + (py - height/2) * pixSize;
            
            int r = 0, g = 0, b = 0;
            //supersampling!
            int i;
            int samples = 4;
            for (i=0; i<samples; i++) {
                double dx = (((double)rand() / RAND_MAX) - 0.5) * pixSize;
                double dy = (((double)rand() / RAND_MAX) - 0.5) * pixSize;
                colour c = drawMandelbulb(cx+dx, cy+dy);
                r += c.r;
                g += c.g;
                b += c.b;
            }
            
            data[py][px].r = r/samples;
            data[py][px].g = g/samples;
            data[py][px].b = b/samples;
        }
    }
    
    writeBMP(outFile, (colour*)data, width, height);

}

colour drawMandelbrot(double x, double y) {
    const int iterations = 256;
    
    double zreal = 0, zimag = 0;
    double tmp;
    
    colour ret;
    
    int i;
    for (i=0; i<iterations && zreal*zreal+zimag*zimag < 2*2; i++) {
        tmp = zreal*zreal - zimag*zimag + x;
        zimag = 2 * zreal * zimag + y;
        zreal = tmp;
    }
    
    ret.r = ret.g = ret.b = i;
    
    return ret;
}

colour drawMandelbulb(double x, double y) {
    const double fov = M_PI / 2;
    const double zdist = 0.5 / tan(fov / 2);
    
    
    colour ret;
    
    vec3 eye, direction;
    
    //set the eye position. 'down' and 'back' from the fractal
    eye.x = 0;
    eye.y = 3;
    eye.z = -3;
    
    //set the view direction. looking up at the fractal
    direction.x = x;
    direction.y = y;
    direction.z = zdist;
    vecNormalizeInplace(&direction);
    vecRotateXInplace(&direction, M_PI/4);
    
    raymarchResult v = mandelbulbRaymarch(eye, direction);
    if (v.hit) {
        vec3 normal = mandelbulbEstimateNormal(v.pos);
        
        ret.r = (normal.y+1)/2 * 255;
        ret.g = (normal.x+1)/2 * 255;
        ret.b = (normal.z+1)/2 * 255;
    } else {
        ret.r = 0;
        ret.g = 0;
        ret.b = 0;
    }
    

    return ret;
}

double mandelbulbDistanceEstimator(vec3 pos) {
    //estimate the distance to the fractal
    //from a position.
    //This is essential for rending the fractal
    //through raymarching, as we can just keep marching
    //forward based on this estimation, eventually
    //converging on the edge of the fractal
    
    //we run a few iterations of the fractal
    //keeping track of the 'running derivitive'
    //of the magnitude, and pass that to a magical
    //formula
    
    
    const int iterations = 13;
    const int power = 8;
    const int bailout = 4;
    
    vec3 z = pos;
    double magnitude = 0;
    double dmagnitude = 1; //running derivitive of magnitude
    double newMagnitude;
    
    
    int i;
    for (i=0; i < iterations; i++) {
        magnitude = vecLength(z);
        if (magnitude > bailout) {
            break;
        }
        
        //make z polar so we can exponent easilly
        double theta = acos(z.z / magnitude);
        double phi = atan2(z.y, z.x);
        
        dmagnitude = pow(magnitude, power-1) * power * dmagnitude + 1;
        
        //now exponent
        newMagnitude = pow(magnitude, power);
        theta *= power;
        phi *= power;
        
        //project back to cartesian using mandelbulb projection
        z.x = newMagnitude*sin(theta)*cos(phi);
        z.y = newMagnitude*sin(theta)*sin(phi);
        z.z = newMagnitude*cos(theta);
        
        //and add 'c'
        z.x += pos.x;
        z.y += pos.y;
        z.z += pos.z;
    }
    
    //printf("(%lf,%lf)", magnitude, dmagnitude); 
    
    return 0.5 * log(magnitude) * magnitude / dmagnitude;
    
}

vec3 mandelbulbEstimateNormal(vec3 pos) {
    const double epsilon = 1e-7;
    
    vec3 ret;
    
    //
    ret.x = mandelbulbDistanceEstimator(vecAddX(pos, epsilon))
            - mandelbulbDistanceEstimator(vecAddX(pos, -epsilon));
            
    ret.y = mandelbulbDistanceEstimator(vecAddY(pos, epsilon))
            - mandelbulbDistanceEstimator(vecAddY(pos, -epsilon));
    
    ret.z = mandelbulbDistanceEstimator(vecAddZ(pos, epsilon))
            - mandelbulbDistanceEstimator(vecAddZ(pos, -epsilon));
    
    vecNormalizeInplace(&ret);
    
    return ret;
}

raymarchResult mandelbulbRaymarch(vec3 pos, vec3 direction) {
    const int maxSteps = 128;
    const double maxDist = 4;
    const double minDist = 0.0005;
    
    double stepDistance;
        
    raymarchResult ret;
    int step;
    
    ret.hit = FALSE;
    ret.distance = 0;
    for (step = 0; step < maxSteps; step++) {
        stepDistance = mandelbulbDistanceEstimator(pos);
        
        if (stepDistance < minDist) {
            ret.hit = TRUE;
            break;
        }
        if (stepDistance > maxDist) {
            break;
        }
        
        //printf("+%lf_%lf ", distance, totalDistance);
        
        ret.distance += stepDistance;
        
        pos.x += stepDistance * direction.x;
        pos.y += stepDistance * direction.y;
        pos.z += stepDistance * direction.z;
    }
    
    ret.pos = pos;
    
    return ret;
}


void writeBMP(FILE *outFile, colour* data, int width, int height) {
    bmpHeader header;
    bmpInfoHeader infoHeader;
    int x, y;
    int stride = ((width*3 + 3) / 4) * 4;
    
    //yay gcc automagic memory
    uint8_t bmpData[height][stride];
    
    
    //firstly write the bmp magic
    fwrite("BM", 1, 2, outFile);
    
    //populate the header
    
    //3 bytes per pixel, but row aligned to 4 bytes
    header.fileSize = 2 + sizeof(bmpHeader) + sizeof(bmpInfoHeader)
                        + stride*height;
    
    header.creator1 = 0;
    header.creator2 = 0;
    header.dataOffset = 2 + sizeof(bmpHeader) + sizeof(bmpInfoHeader);
    
    //write the header
    fwrite(&header, sizeof(header), 1, outFile);
    
    //populate the DIB header
    infoHeader.headerSize = sizeof(bmpInfoHeader);
    infoHeader.width = width;
    infoHeader.height = height;
    infoHeader.planes = 1;
    infoHeader.bitsPerPixel = 24;
    
    //bmp compression, so don't need to populate imageSize
    infoHeader.compression = 0;
    infoHeader.imageSize = 0;
    
    infoHeader.xPixelsPerMeter = 2835;
    infoHeader.yPixelsPerMeter = 2835;
    
    //only used for indexed colour palettes
    infoHeader.userColours = 0;
    infoHeader.importantColours = 0;
    
    //write the DIB header
    fwrite(&infoHeader, sizeof(infoHeader), 1, outFile);
    
    //we need to rewrite the image data into the correct stride
    memset(bmpData, 0, height*stride);
    
    for (y=0; y<height; y++) {
        for (x=0; x<width; x++) {
            bmpData[y][x*3+0] = data[y*width+x].b;
            bmpData[y][x*3+1] = data[y*width+x].g;
            bmpData[y][x*3+2] = data[y*width+x].r;
        }
    }
    
    //write the image data
    fwrite(data, 1, height*stride, outFile);
    
    fclose(outFile);
}


double vecLength(vec3 v) {
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

void vecNormalizeInplace(vec3 *v) {
    double mag = vecLength(*v);
    v->x /= mag;
    v->y /= mag;
    v->z /= mag;
}

void vecRotateXInplace(vec3 *v, double theta) {
    vec3 ret;
    ret.x = v->x;
    ret.y = v->y * cos(theta) + v->z * -sin(theta);
    ret.z = v->y * sin(theta) + v->z * cos(theta);
    
    v->x = ret.x;
    v->y = ret.y;
    v->z = ret.z;
}

void vecRotateYInplace(vec3 *v, double theta) {
    vec3 ret;
    ret.x = v->x * cos(theta) + v->z * sin(theta);
    ret.y = v->y;
    ret.z = v->x * -sin(theta) + v->z * cos(theta);
    
    v->x = ret.x;
    v->y = ret.y;
    v->z = ret.z;
}

vec3 vecAddX(vec3 v, double c) {
    v.x += c;
    return v;
}

vec3 vecAddY(vec3 v, double c) {
    v.y += c;
    return v;
}

vec3 vecAddZ(vec3 v, double c) {
    v.z += c;
    return v;
}
