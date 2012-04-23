// lolmandelbulb.c
// comp1917 task2 made awesome
// Created by Niel van der Westhuizen, 09/04/2012

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

#include <signal.h>
#include <netinet/in.h>

#include "pixelColor.h"

#define TRUE 1
#define FALSE 0

//wtf c standard
#ifndef max
   #define max(a,b) (((a) > (b)) ? (a) : (b))
   #define min(a,b) (((a) < (b)) ? (a) : (b))
#endif


#define SERVER_PORT 8080


typedef struct _bmpHeader {
    uint32_t fileSize;
    uint16_t creator1;
    uint16_t creator2;
    uint32_t dataOffset;
} __attribute__((packed)) bmpHeader;

//see http://msdn.microsoft.com/en-us/library/dd183376(v=vs.85).aspx
typedef struct _bmpInfoHeader {
    uint32_t infoHeaderSize; //40
    
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
    uint32_t usedColours;
    uint32_t importantColours;
} __attribute__((packed)) bmpInfoHeader;

typedef struct _colour {
    uint8_t b, g, r;
} colour;

typedef struct _vec3 {
    double x, y, z;
} vec3;


typedef struct _raymarchResult {
    int hit;
    double distance;
    vec3 pos;
} raymarchResult;

void runFractalServer(int portNumber,
                      colour (*drawFunction)(double x, double y));
void writeStringToSocket(int socket, const char* string);
void writeFileToSocket(int socket, const char* fileName);
int waitForConnection(int serverSocket);
int makeServerSocket(int portNumber);


void renderFractal(FILE *outFile,
                   colour (*drawFunction)(double x, double y),
                   int width, int height,
                   int zoom, double x, double y);

void writeBMP(FILE *outFile, colour* data, int width, int height);

void printProgressBar(int width, double progress);


colour drawMandelbrot(double x, double y);
colour drawMandelbulb(double x, double y);

raymarchResult mandelbulbRaymarch(vec3 pos, vec3 direction);
double mandelbulbDistanceEstimator(vec3 pos);
vec3 mandelbulbEstimateNormal(vec3 pos);

int clampRayToSphere(vec3 *rayPos, vec3 rayDirection,
                     vec3 spherePos, double radius);

vec3 phongShade(vec3 pos, vec3 normal, vec3 eye, vec3 light);


double vecLength(vec3 v);
void vecNormalizeInplace(vec3 *v);
vec3 vecNormalize(vec3 v);
void vecRotateXInplace(vec3 *v, double theta);
void vecRotateYInplace(vec3 *v, double theta);
vec3 vecAddX(vec3 v, double c);
vec3 vecAddY(vec3 v, double c);
vec3 vecAddZ(vec3 v, double c);
vec3 vecAdd(vec3 a, vec3 b);
void vecAddInplace(vec3 *a, vec3 b);
vec3 vecAdd3(vec3 a, vec3 b, vec3 c);
vec3 vecSub(vec3 a, vec3 b);
vec3 vecMul(vec3 a, double c);
void vecMulInplace(vec3 *a, double c);
double vecDot(vec3 a, vec3 b);

double randd();

int main(int argc, char **argv) {
    FILE *outFile;
    int zoom = 7;
    double x = 0, y = 0;
    int width = 512, height = 512;

    int port = SERVER_PORT;

    colour (*drawFunction)(double x, double y) = drawMandelbulb;

    int successfulArguments = TRUE;

    int optionCharacter;
    while (TRUE) {
        optionCharacter = getopt(argc, argv, "f:p:o:x:y:z:w:h:");
        if (optionCharacter == -1) {
            break;
        }

        //not allowed to use switch...
        if (optionCharacter == 'f') {
            if (strcmp(optarg, "mandelbrot") == 0) {
                drawFunction = drawMandelbrot;
            } else if (strcmp(optarg, "mandelbulb") == 0) {
                drawFunction = drawMandelbulb;
            } else {
                fprintf(stderr,
                  "Valid fractals are 'mandelbrot' and 'madelbulb'\n");
                return 1;
            }
        } else if (optionCharacter == 'p') {
            port = atoi(optarg);
        } else if (optionCharacter == 'o') {
            outFile = fopen(optarg, "wb");
            assert(outFile);
        } else if (optionCharacter == 'x') {
            x = strtod(optarg, NULL);
        } else if (optionCharacter == 'y') {
            y = strtod(optarg, NULL);
        } else if (optionCharacter == 'z') {
            zoom = atoi(optarg);
        } else if (optionCharacter == 'w') {
            width = atoi(optarg);
        } else if (optionCharacter == 'h') {
            height = atoi(optarg);
        } else {
            successfulArguments = FALSE;
        }
    }

    if (optind < argc) {
        successfulArguments = FALSE;
    }

    if (!successfulArguments) {
        fprintf(stderr, "Usage: %s [-f fractal] [-p serverport] "
                    "[-o outfile [-x x] [-y y] "
                      "[-z zoom] [-w width] [-h height]]\n",
                    argv[0]);
    } else if (outFile) { //asked to render to a file
        renderFractal(outFile, drawFunction,
              width, height, zoom, x, y);

        fclose(outFile);
    } else {
        runFractalServer(port, drawFunction);
    }
    
    return 0;
}


void runFractalServer(int portNumber,
                      colour (*drawFunction)(double x, double y)) {
    const int width = 512;
    const int height = 512;
    
    int zoom;
    double x, y;
    
    int connectionSocket;
    int bytesRead;
    
    char requestBuffer[1024];
    char requestUrl[1024];
    
    int serverSocket = makeServerSocket(portNumber);
    
    printf("Serving fractals on http://localhost:%d\n", portNumber);
    
    while (TRUE) {
        connectionSocket = waitForConnection(serverSocket);
        
        bytesRead = read(connectionSocket, requestBuffer,
                             sizeof(requestBuffer)-1);
        
        //printf("http request: %s\n", requestBuffer);
        
        if (sscanf(requestBuffer, "GET %s HTTP/%*d.%*d\r\n",
                   requestUrl) == 1) {
            
            //we have a request!
            printf("GET %s\n", requestUrl);
            
            if (sscanf(requestUrl,
                       "/X-%lf-%lf-%d.bmp", &x, &y, &zoom) == 3) {
                //we're requested to serve a render!
                
                writeStringToSocket(connectionSocket,
                                    "HTTP/1.0 200 OK\r\n"
                                    "Content-Type: image/x-ms-bmp\r\n"
                                    "\r\n");
                
                
                //open the socket as a file handle so we can reuse
                //renderFractal code for files :D
                FILE *responseHandle = fdopen(connectionSocket, "w+");
                
                if (responseHandle) {
                    renderFractal(responseHandle, drawFunction,
                                  width, height, zoom, x, y);

                    fflush(responseHandle);
                    //assert(fclose(responseHandle) == 0);
                }
            } else if (strcmp(requestUrl, "/tile.min.js") == 0) {
                //send the interface logic javascript
                writeStringToSocket(connectionSocket,
                                    "HTTP/1.0 200 Found\r\n"
                                    "Content-Type: text/javascript\r\n"
                                    "\r\n");
                
                writeFileToSocket(connectionSocket, "tile.min.js");
            } else {
                //any other request, serve the viewer html
                writeStringToSocket(connectionSocket,
                                    "HTTP/1.0 200 Found\r\n"
                                    "Content-Type: text/html\r\n"
                                    "\r\n"
                                    "<html>"
                                    "<script src=\""
                                    "/tile.min.js"
//"https://openlearning.cse.unsw.edu.au/site_media/viewer/tile.min.js"
                                    
                                    "\"></script></html>");
            }
        } else {
            //did not recieve a valid HTTP get
            writeStringToSocket(connectionSocket,
                                "HTTP/1.0 400 Bad Request\r\n");
        }
        
        //assert(close(connectionSocket) == 0);
        close(connectionSocket);
    }
}

void writeStringToSocket(int socket, const char* string) {
    assert(write(socket,
                 string, strlen(string))
            == strlen(string));
}

void writeFileToSocket(int socket, const char* fileName) {
    FILE *file = fopen(fileName, "r");
    uint8_t sendBuffer[1024];
    size_t read;
    
    assert(file);
    while (!feof(file)) {
        read = fread(sendBuffer, 1, sizeof(sendBuffer), file);
        assert( write(socket, sendBuffer, read) == read );
    }
    fclose(file);
}

//start the server listening on the specified port number
//stolen from comp1917 "simpleServer.c"
int makeServerSocket(int portNumber) { 
    int serverSocket;
    struct sockaddr_in serverAddress;
    int reuseAddressValue = 1;
    int bindSuccess;
    
    //don't crash on SIGPIPE
    //if the socket breaks we don't /really/ care
    signal(SIGPIPE, SIG_IGN);
    
    //create socket
    serverSocket = socket (AF_INET, SOCK_STREAM, 0);
    
    //ensure no error opening socket
    assert (serverSocket >= 0);   
    
    // bind socket to listening port
    bzero ((char *) &serverAddress, sizeof (serverAddress));
    
    serverAddress.sin_family      = AF_INET;
    serverAddress.sin_addr.s_addr = INADDR_ANY;
    serverAddress.sin_port        = htons (portNumber);
    
    // let the server start immediately after a previous shutdown
    setsockopt(serverSocket,
              SOL_SOCKET,
              SO_REUSEADDR,
              &reuseAddressValue, 
              sizeof(int));
    
    bindSuccess = bind(serverSocket, 
                       (struct sockaddr *)&serverAddress,
                       sizeof (serverAddress));
    
    // if this assert fails wait a short while to let the operating 
    // system clear the port before trying again
    assert (bindSuccess >= 0);
    
    return serverSocket;
}

//wait for a browser to request a connection,
//returns the socket on which the conversation will take place
//stolen from comp1917 "simpleServer.c"
int waitForConnection(int serverSocket) {
    // listen for a connection
    const int serverMaxBacklog = 10;
    
    int connectionSocket;
    struct sockaddr_in clientAddress;
    socklen_t clientLen;
    
    listen (serverSocket, serverMaxBacklog);
    
    // accept the connection
    clientLen = sizeof(clientAddress);
    connectionSocket = accept(
        serverSocket, 
        (struct sockaddr *)&clientAddress, 
        &clientLen);
    
    // ensure no error on accept
    assert (connectionSocket >= 0);
    
    return connectionSocket;
}

void renderFractal(FILE *outFile,
                   colour (*drawFunction)(double x, double y),
                   int width, int height,
                   int zoom, double x, double y) {

    const int supersamplingSamples = 4;
    
    int px, py; //pixel location
    double cx, cy; //pixel location on the draw plane
    double pixelSize = pow(2, -zoom); //pixel size on the draw plane
    colour data[height][width];
    
    int i;
    double dx, dy; //supersample offsets
    colour initialColour, sampleColour;
    int cumr, cumg, cumb; //cumulative sum colours for supersampling
    
    int completedRows = 0; //for progress bar
    
    
#pragma omp parallel for shared(data, completedRows) \
                         private(px, py, cx, cy, \
                                 i, dx, dy, \
                                 initialColour, \
                                 sampleColour, \
                                 cumr, cumg, cumb)
    for (py=0; py<height; py++) {
        for (px=0; px<width; px++) {
            //map the pixel onto the complex plane
            cx = x + (px - width/2) * pixelSize;
            cy = y - (py - height/2) * pixelSize;
            
            //sample the colour the first time seperately
            //for handling no supersampling
            initialColour = drawFunction(cx, cy);
            cumr = initialColour.r;
            cumg = initialColour.g;
            cumb = initialColour.b;
            
            //supersampling!
            //sample the rest with random diviations from the centre.
            for (i=1; i<supersamplingSamples; i++) {
                dx = (randd() - 0.5) * pixelSize;
                dy = (randd() - 0.5) * pixelSize;
                
                sampleColour = drawFunction(cx+dx, cy+dy);
                
                cumr += sampleColour.r;
                cumg += sampleColour.g;
                cumb += sampleColour.b;
            }
            
            data[py][px].r = cumr / supersamplingSamples;
            data[py][px].g = cumg / supersamplingSamples;
            data[py][px].b = cumb / supersamplingSamples;
        }

        #pragma omp critical
        {
            completedRows++;
            printProgressBar(80, (double)completedRows/height);
        }
    }
    printf("\n");
    
    writeBMP(outFile, (colour*)data, width, height);

}

void writeBMP(FILE *outFile, colour* data, int width, int height) {
    bmpHeader header;
    bmpInfoHeader infoHeader;
    int x, y;
    //24-bits per pixel, rounded up to 32-bit (4-byte) boundary
    int stride = ((24*width+31)/32)*4;
    
    //yay gcc automagic memory
    uint8_t bmpData[height][stride];
    
    //firstly write the bmp magic
    fwrite("BM", 1, 2, outFile);
    
    //populate the header
    header.fileSize = 2 + sizeof(bmpHeader) + sizeof(bmpInfoHeader)
                        + stride*height;
    header.creator1 = 0;
    header.creator2 = 0;
    header.dataOffset = 2 + sizeof(bmpHeader) + sizeof(bmpInfoHeader);
    
    //write the header
    fwrite(&header, sizeof(header), 1, outFile);
    
    //populate the DIB header
    infoHeader.infoHeaderSize = sizeof(bmpInfoHeader);
    infoHeader.width = width;
    //we set a negative height as normally the data has the be
    //bottom-to-top.
    infoHeader.height = -height;
    infoHeader.planes = 1;
    infoHeader.bitsPerPixel = 24;
    
    //no compression. don't need to populate imageSize
    infoHeader.compression = 0;
    infoHeader.imageSize = 0;
    
    infoHeader.xPixelsPerMeter = 2835;
    infoHeader.yPixelsPerMeter = 2835;
    
    //only used for indexed colour palettes
    infoHeader.usedColours = 0;
    infoHeader.importantColours = 0;
    
    //write the DIB header
    fwrite(&infoHeader, sizeof(infoHeader), 1, outFile);
    
    //we need to rewrite the image data into the correct stride
    //and format
    //24-bit bmp is "blue, green and red, 8-bits per each sample"
    bzero(bmpData, height*stride);
    for (y=0; y<height; y++) {
        for (x=0; x<width; x++) {
            bmpData[y][x*3+0] = data[y*width+x].b;
            bmpData[y][x*3+1] = data[y*width+x].g;
            bmpData[y][x*3+2] = data[y*width+x].r;
        }
    }
    
    //write the image data
    fwrite(bmpData, 1, height*stride, outFile);
}

void printProgressBar(int width, double progress) {
    int endPos = width-5;
    int x;
    
    //ansi control code to position cursor to column 0
    printf("\x1b[1G");
    
    printf("[");
    for (x=1; x<endPos*progress; x++) {
        printf("=");
    }
    for (; x<endPos; x++) {
        printf(" ");
    }
    printf("]");
    //print the percentage
    printf(" %02d%%", min((int)(progress*100), 99));
    fflush(stdout);
}

colour drawMandelbrot(double x, double y) {
    const int iterations = 4096;
    const int bailout = 16;
    const int smoothColouring = TRUE;

    //make points in the set black;
    const colour setColour = {0, 0, 0};
    
    double zreal = 0, zimag = 0;
    double tmp;

    //for smooth colouring
    double zMagnitude;
    double iReal;
    double fraction;
    int sample;

    
    colour ret;
    
    int i;
    for (i=0; i<iterations
          && zreal*zreal+zimag*zimag < bailout*bailout; i++) {
        //z = z^2 + (x + yi)
        
        tmp = zreal*zreal - zimag*zimag + x;
        zimag = 2 * zreal * zimag + y;
        zreal = tmp;
    }

    //colour the pixel!
    if (i >= iterations) {
        //point is in the set (so should be black)
        ret = setColour;
    } else if (smoothColouring) {
        zMagnitude = hypot(zreal, zimag);

        //See Wikipedia:Mandelbrot_set#Continuous_.28smooth.29_coloring
        iReal = i-log2(log(zMagnitude));
        if (iReal < 0) {
            iReal = 0;
        }

        //Since the pixelColor.h header is retarded and
        //only takes integers, take the colour either
        //side and interpolate them
        sample = floor(iReal)-1;
        fraction = 1-fmod(iReal, 1);

        ret.r = min(max(stepsToRed(sample)*fraction
                 + stepsToRed(sample+1)*(1-fraction), 0), 255);
        ret.g = min(max(stepsToGreen(sample)*fraction
                 + stepsToGreen(sample+1)*(1-fraction), 0), 255);
        ret.b = min(max(stepsToBlue(sample)*fraction
                 + stepsToBlue(sample+1)*(1-fraction), 0), 255);
    } else {
        ret.r = stepsToRed(i);
        ret.g = stepsToGreen(i);
        ret.b = stepsToBlue(i);
    }
    
    
    return ret;
}

colour drawMandelbulb(double x, double y) {
    const double fov = M_PI / 2;
    const double zdist = 0.5 / tan(fov / 2);
    
    const vec3 light = {30, 30, -40};
    const int drawShadow = TRUE;
    
    //values used for lighting
    vec3 normal;
    vec3 shading;
    vec3 dirFromLight;
    vec3 tmpColouring;
    
    raymarchResult primaryrayResult, shadowRayResult;
    
    colour ret;
    
    //set the eye position. 'down' and 'back' from the fractal
    vec3 eye = {0, 3, -3};
    
    //set the view direction - looking up at the fractal
    vec3 direction = {x, y, zdist};
    vecNormalizeInplace(&direction);
    vecRotateXInplace(&direction, M_PI/4);
    
    primaryrayResult = mandelbulbRaymarch(eye, direction);
    
    if (primaryrayResult.hit) {
        normal = mandelbulbEstimateNormal(primaryrayResult.pos);

        shading = phongShade(primaryrayResult.pos, normal, eye, light);
        
        if (drawShadow) {
            //cast 'shadow ray' from the light to the point.
            //Opposite to the traditional point-to-light shadow
            //cast because we can't cast reliably away from the
            //fractal using the distance estimation method.
            dirFromLight = vecNormalize(vecSub(primaryrayResult.pos,
                                            light));
            shadowRayResult = mandelbulbRaymarch(light, dirFromLight);
            
            //if the shadow ray isn't close to the render point,
            //then it hit something along the way so we're in
            //shadow.
            if (vecLength(vecSub(shadowRayResult.pos,
                                primaryrayResult.pos)) > 0.01) {
                vecMulInplace(&shading, 0.5);
            }
        }
        
        //use the normal for some interesting surface detail
        tmpColouring = vecMul((vec3){1, 1, 1},
                          (normal.x+normal.y+normal.z+3)/3);
        
        tmpColouring.x *= shading.x;
        tmpColouring.y *= shading.y;
        tmpColouring.z *= shading.z;
        
        ret.r = min(max(tmpColouring.x, 0), 1) * 255;
        ret.g = min(max(tmpColouring.y, 0), 1) * 255;
        ret.b = min(max(tmpColouring.z, 0), 1) * 255;
        
    } else {
        //missed the fractal, set the background colour to
        //slightly off-white
        ret.r = 240;
        ret.g = 240;
        ret.b = 240;
    }

    return ret;
}

raymarchResult mandelbulbRaymarch(vec3 pos, vec3 direction) {
    const int maxSteps = 512;
    const double maxStep = 4;
    const double minStep = 0.0001;
    
    double stepDistance;
        
    raymarchResult ret;
    int step;
    
    //the mandelbulb is inside a radius ~1.5 sphere
    //distance estimation breaks down if we're too far away
    //so start the ray at the intersection point with the
    //sphere
    int hitsSphere = clampRayToSphere(&pos, direction,
                                      (vec3){0, 0, 0}, 1.5);
    
    ret.hit = FALSE;
    ret.distance = 0;
    if (hitsSphere) {
        for (step = 0; step < maxSteps; step++) {
            stepDistance = mandelbulbDistanceEstimator(pos);
            
            if (stepDistance < minStep) {
                ret.hit = TRUE;
                break;
            }
            
            //we're likely just getting further from the fractal, abort
            if (stepDistance > maxStep) {
                break;
            }
            
            ret.distance += stepDistance;
            
            pos.x += stepDistance * direction.x;
            pos.y += stepDistance * direction.y;
            pos.z += stepDistance * direction.z;
        }
    }
    ret.pos = pos;
    
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
    
    //Math stolen from 'Mikael Hvidtfeldt Christensen'
    //http://blog.hvidtfeldts.net
    
    
    const int iterations = 30;
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
        
        dmagnitude = pow(magnitude, power-1) * power * dmagnitude + 1;
        
        
        //make z polar so we can exponent easilly
        double theta = acos(z.z / magnitude);
        double phi = atan2(z.y, z.x);
        
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
    
    return 0.5 * log(magnitude) * magnitude / dmagnitude;
}

vec3 mandelbulbEstimateNormal(vec3 pos) {
    const double epsilon = 1e-7;
    
    vec3 ret;
    
    //So the normal is akin to the derivative of the surface of the
    //fractal. Estimate it by taking the difference of 'distances'
    //sampled either side of the point in each axis.
    ret.x = mandelbulbDistanceEstimator(vecAddX(pos, epsilon))
             - mandelbulbDistanceEstimator(vecAddX(pos, -epsilon));
    
    ret.y = mandelbulbDistanceEstimator(vecAddY(pos, epsilon))
             - mandelbulbDistanceEstimator(vecAddY(pos, -epsilon));
    
    ret.z = mandelbulbDistanceEstimator(vecAddZ(pos, epsilon))
             - mandelbulbDistanceEstimator(vecAddZ(pos, -epsilon));
    
    vecNormalizeInplace(&ret);
    
    return ret;
}


//returns false if the ray misses the sphere
//based on http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
int clampRayToSphere(vec3 *rayPos, vec3 rayDirection,
                     vec3 spherePos, double radius) {
    
    //compute line coefficents from ray
    vec3 rayToSphere = vecSub(*rayPos, spherePos);
    double b = vecDot(rayToSphere, rayDirection);
    double c = vecDot(rayToSphere, rayToSphere) - radius * radius;
    
    double discriminant = b * b - c;
    double dist;
    
    int hit = TRUE;
    if (discriminant < 0) {
        hit = FALSE;
    } else {
        
        //distance to point of closest intersection
        dist = -b - sqrt(discriminant);
        
        //advance the ray to it
        vecAddInplace(rayPos, vecMul(rayDirection, dist));
    }
    
    return hit;
}

//computes phong shading
//returns a colour in the form form of a vec3 for easier mixing
//https://en.wikipedia.org/wiki/Phong_reflection_model
vec3 phongShade(vec3 pos, vec3 normal, vec3 eye, vec3 light) {
    
    const vec3 ambientColour = {0.31, 0.19, 0.055};
    const vec3 diffuseColour = {0.61, 0.48, 0.20};
    const vec3 specularColour = {1, 1, 1};
    const double specularity = 0.4;
    const double shininess = 1;
    
    vec3 lightDirection;
    vec3 eyeDirection;
    double normalDotLight;
    vec3 reflected;
    double reflectionDotEye;
    double reflectionSpecular;
    
    vec3 diffuse = {0, 0, 0};
    vec3 specular = {0, 0, 0};
    vec3 resColour;
    
    
    lightDirection = vecSub(light, pos);
    vecNormalizeInplace(&lightDirection);
    
    eyeDirection = vecSub(eye, pos);
    vecNormalizeInplace(&eyeDirection);
    
    normalDotLight = vecDot(normal, lightDirection);
    
    if (normalDotLight > 0) { //if surface is facing towards the light
        //diffuse shading
        diffuse.x = diffuseColour.x * normalDotLight;
        diffuse.y = diffuseColour.y * normalDotLight;
        diffuse.z = diffuseColour.z * normalDotLight;
        
        //specular highlight
        //find the reflected vector
        reflected.x = lightDirection.x - 2 * normalDotLight * normal.x;
        reflected.z = lightDirection.y - 2 * normalDotLight * normal.y;
        reflected.y = lightDirection.z - 2 * normalDotLight * normal.z;
        
        reflectionDotEye = vecDot(reflected, eyeDirection);
        
        //if the reflection is in the direction of the eye
        if (reflectionDotEye <= 0) {
            //calculate some specular
            reflectionSpecular = specularity * pow(
                fabs(reflectionDotEye), shininess);
            specular = vecMul(specularColour, reflectionSpecular);
        }
    }
    
    resColour = vecAdd3(ambientColour, diffuse, specular);
    
    //clamp everything down into 0-1
    resColour.x = min(max(resColour.x, 0), 1);
    resColour.y = min(max(resColour.y, 0), 1);
    resColour.z = min(max(resColour.z, 0), 1);
    
    return resColour;
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

vec3 vecNormalize(vec3 v) {
    vecNormalizeInplace(&v);
    return v;
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

vec3 vecAdd(vec3 a, vec3 b) {
    vec3 ret;
    ret.x = a.x + b.x;
    ret.y = a.y + b.y;
    ret.z = a.z + b.z;
    return ret;
}

void vecAddInplace(vec3 *a, vec3 b) {
    a->x += b.x;
    a->y += b.y;
    a->z += b.z;
}

vec3 vecAdd3(vec3 a, vec3 b, vec3 c) {
    vec3 ret;
    ret.x = a.x + b.x + c.x;
    ret.y = a.y + b.y + c.y;
    ret.z = a.z + b.z + c.z;
    return ret;
}

vec3 vecSub(vec3 a, vec3 b) {
    vec3 ret;
    ret.x = a.x - b.x;
    ret.y = a.y - b.y;
    ret.z = a.z - b.z;
    return ret;
}

vec3 vecMul(vec3 a, double c) {
    vec3 ret;
    ret.x = a.x * c;
    ret.y = a.y * c;
    ret.z = a.z * c;
    return ret;
}

void vecMulInplace(vec3 *a, double c) {
    a->x *= c;
    a->y *= c;
    a->z *= c;
}

double vecDot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

//returns a random double between 0.0 and 1.0
double randd() {
    return (double)rand() / RAND_MAX;
}