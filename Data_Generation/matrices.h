/*
 * matrices.h
 *
 *  Created on: Jun 19, 2015
 *      Author: jahnka
 */

#ifndef MATRICES_H_
#define MATRICES_H_

int ** transposeMatrix(int** matrix, int n, int m);
void addToMatrix(int** first, int** second, int n, int m);
double** allocate_doubleMatrix(int n, int m);
int** allocate_intMatrix(int n, int m);
bool** allocate_boolMatrix(int n, int m);
double** init_doubleMatrix(int n, int m, double value);
int* init_intArray(int n, int value);
bool* init_boolArray(int n, bool value);
double* init_doubleArray(int n, double value);
int** init_intMatrix(int n, int m, int value);
bool** init_boolMatrix(int n, int m, bool value);
void reset_intMatrix(int** matrix, int n, int m, int value);
void free_boolMatrix(bool** matrix);
void free_intMatrix(int** matrix);
void free_doubleMatrix(double** matrix);
void delete_3D_intMatrix(int*** matrix, int n);
double getMaxEntry(double* array, int n);
bool** deepCopy_boolMatrix(bool** matrix, int n, int m);
int** deepCopy_intMatrix(int** matrix, int n, int m);
int* deepCopy_intArray(int* array, int n);
void print_boolMatrix(bool** array, int n, int m);
void print_doubleMatrix(double** matrix, int n, int m);
void print_intMatrix(int** matrix, int n, int m, char del);
void print_intArray(int* array, int n);
int* ancMatrixToParVector(bool** anc, int n);



#endif /* MATRICES_H_ */
