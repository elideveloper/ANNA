#ifndef UTILITY_H
#define UTILITY_H

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

inline double boundBetween(double min, double max, double val)
{
	if (val < min) return min;
	if (val > max) return max;
	return val;
}

#endif