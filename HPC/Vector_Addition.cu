/*
This code contains comparison using time for CPU And GPU.
*/

#include<bits/stdc++.h>
using namespace std;

void initialize(int *vector,int size){
    for(int i=0;i<size;i++){
        vector[i] = rand()%10;
    }
}

void print(int *vector,int size){
  for(int i=0;i<size;i++){
      cout<<vector[i] << " ";
  }
  cout<<endl;
}

__global__ void add_cpu(int *A, int *B,int *C,int size){
    for(int i=0;i<size;i++){
        C[i] = A[i] + B[i];
    }
}

__global__ void add_gpu(int *A, int *B,int *C,int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size){
        C[tid] = A[tid] + B[tid];
    }
}

int main(){
    int N = 5;
    int vectorSize = N;
    size_t vectorBytes= vectorSize * sizeof(int);

    int *A,*B,*C;

    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];

    initialize(A,N);
    initialize(B,N);

    cout<<"Vector A : ";
    print(A,N);

    cout<<"Vector B : ";
    print(B,N);

    int *X,*Y,*Z;
    cudaMalloc(&X,vectorBytes);
    cudaMalloc(&Y,vectorBytes);
    cudaMalloc(&Z,vectorBytes);

    cudaMemcpy(X,A,vectorBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(Y,B,vectorBytes,cudaMemcpyHostToDevice);
    
    cudaEvent_t start,stop;
    float elapsedTime;

    int threadsPerBlock = 256;
    int blocksPerGrid = N + threadsPerBlock -1 / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    add_gpu<<<blocksPerGrid, threadsPerBlock>>>(X,Y,Z,N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(C,Z,vectorBytes,cudaMemcpyDeviceToHost);
    cout<<"GPU RESULT : ";
    print(C,N); 
    cout<<"Elapsed Time : "<<elapsedTime<<endl;

    //cpu
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    add_cpu<<<blocksPerGrid, threadsPerBlock>>>(X,Y,Z,N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(C,Z,vectorBytes,cudaMemcpyDeviceToHost);
    cout<<"CPU RESULT : ";
    print(C,N); 
    cout<<"Elapsed Time : "<<elapsedTime<<endl;

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);
    return 0;
}
