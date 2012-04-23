// pixelColor.c
// comp1917 task2 made awesome
// Created by Niel van der Westhuizen 20/04/2012

#include <math.h>
#include "pixelColor.h"

//So we colour the mandelbrot with colour maps
//based on gradients which we specify with some splines
//for easy tweaking :D

const double gradientScale = 0.056;
const double gradientOffset = 0;

double redGradientPoints[][2] = {
    {0.0,  0.0},
    {0.16, 1.0},
    {0.5,  0.0},
    {0.83, 1.0},
    {1.0,  0.0}
};

double greenGradientPoints[][2] = {
    {0.0,  0.0},
    {0.25, 1.0},
    {0.5,  0.0},
    {0.66, 1.0},
    {1.0,  0.0}
};

double blueGradientPoints[][2] = {
    {0.0,  0.0},
    {0.33, 1.0},
    {0.5,  0.0},
    {0.75, 1.0},
    {1.0,  0.0}
};

//put the steps into the range of the gradient based on
//the gradient scale and offset
double normaliseSteps(int steps) {
    return fmod(gradientScale * steps + 2 * gradientOffset, 1);
}

//from http://paulbourke.net/miscellaneous/interpolation/
double cosineInterpolate(double v1, double v2, double mu) {
    double mu2;

    mu2 = (1-cos(mu*M_PI))/2;
    return (v1*(1-mu2) + v2*mu2);
}


double mapGradient(double x, double points[][2], int numPoints) {
    int i;
    double ret = points[numPoints-1][1];
    double ratioBetweenPoints;
    
    for (i=1; i<numPoints; i++) {
        if (points[i][0] >= x) {
            ratioBetweenPoints = (x-points[i-1][0])
                                  /(points[i][0]-points[i-1][0]);
            ret = cosineInterpolate(points[i-1][1],
                points[i][1],
                ratioBetweenPoints);
            break;
        }
    }

    if (ret < 0) {
        ret = 0;
    }
    if (ret > 1) {
        ret = 1;
    }

    return ret;
}

unsigned char stepsToRed (int steps) {
    return mapGradient(normaliseSteps(steps),
        redGradientPoints, 5)*255;
}
unsigned char stepsToBlue (int steps) {
    return mapGradient(normaliseSteps(steps),
        blueGradientPoints, 5)*255;
}
unsigned char stepsToGreen (int steps) {
    return mapGradient(normaliseSteps(steps),
        greenGradientPoints, 5)*255;
}