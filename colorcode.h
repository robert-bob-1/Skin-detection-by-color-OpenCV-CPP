#pragma once

void computeColor(float u, float v, unsigned char* color);
void makeColorwheel();

/* 
makeColorwheel() is used for internal initializations; should be called once per application, before any call of computeColor.
computeColor is called for each point. (u, v) represents the displacement vector and "color" stores the output; "color" is a vector with 3 components (r, g, b);
*/
