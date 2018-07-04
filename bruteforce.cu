// in this code we do not use sparsity because the matrices are small any way
// all matrices therefore are full. This actually makes the code faster
// we assume that all the matrices are stored in a linear array, columns first


#include<stdio.h>
#include <time.h>
#include<float.h>

#define MAX_N_PERM 12

typedef unsigned int lint;


#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}



// the code will allow us to use different norms
// the matrices are assume square
// the norm takes as input a permutation as well as two matrices
// do not forget that P is from 1...n while the indices in C are from 0 to n-1
__device__ __host__ float fro_norm_square(int n, float *A , float *B, int *perm){

    float total = 0;
    for (int i = 0; i < n; i++){ //go along column
        for (int j = 0; j < n; j++){ // go along row
            float value = A[i + n*j] - B[ (perm[i]-1) + n*(perm[j]-1)];
            //printf("-- (% f , %f, %f, i = %d,  perm_i = %d,j =%d , perm_j = %d)-- ", A[i + n*j],B[  (perm[i]-1) + n*(perm[j]-1) ],value, i, perm[i]-1, j, perm[j]-1);
            total = total + value*value;
        }
    }
    return total;
}
__device__ float (*d_ptr_fro_norm_square)(int , float * , float *, int *) = fro_norm_square;


// notice that if the matrices are just adjecency matrices where all the weights are 1 then there is no difference between
// having this L1_norm or the fro_norm_square above
__device__ __host__ float L1_norm(int n, float *A , float *B, int *perm){

    float total = 0;
    for (int i = 0; i < n; i++){ //go along column
        for (int j = 0; j < n; j++){ // go along row
            float value = A[i + n*j] - B[ (perm[i]-1) + n*(perm[j]-1)];
            total = total + abs(value);
        }
    }
    return total;
}
__device__ float (*d_ptr_L1_norm)(int , float * , float *, int *) = L1_norm;




// this function sets a list to consecutive numbers
__host__ __device__ inline void settoconsec(int *v, int n){
    for (int i = 0; i < n; i++){
        v[i] = i+1;
    }
}

inline void copyvec(int *v_dest, int * v_source, int n){
    for (int i = 0; i < n; i++){
        v_dest[i] = v_source[i];
    }
}


// this function swaps two elelents
__host__ __device__ inline void swap(int *v, int ix1, int ix2){
    int tmp = v[ix1];
    v[ix1] = v[ix2];
    v[ix2] = tmp;
}

// this function prints a vector
__host__ __device__ void printvec(int *v, int n){
    for (int i = 0; i < n ; i ++){
        printf("%d ", v[i]);
    }
    printf("\n");
}

// this function prints a vector
__host__ __device__ void printfloatvec(float *v, int n){
    for (int i = 0; i < n ; i ++){
        printf("%f ", v[i]);
    }
    printf("\n");
}


// first we have a piece of code that computes the thing in serial form
// we assume that the matrices A and B are square and of the same dimension
// we do not use sparse matrices because the matrices are small anyway
// the norm we are computing is min_P || A - P^T B P ||
// we will also return the best permutation
float compute_optimal_match(int n, float *A, float *B, float (*metric)(int , float* , float *, int* ), int * bestperm ){
    
    float opt_val = FLT_MAX;
    
    int * v = (int *) malloc(n * sizeof(int) );
    int * output = (int *) malloc(n * sizeof(int) );
    
    settoconsec(v, n);
    
    while( v[n-1] <= n ){
        
        // note that the way we are going the swap here is a bit different because
        // the elements from v are already in increasing form. Like
        // 1 2 3 , 2 2 3  , 3 2 3, 1 3 3 , 2 3 3 , 3 3 3
        // while in the parallel code the v is in the form
        // 1 1 1 , 2 1 1, 3 1 1, 1 2 1, 2 2 1 , 3 2 1
        
        settoconsec(output, n);
        for( int i = 0; i < n ; i++){
            swap(output, i, v[i]-1);
        }
        
        // at this point the vector output contains a permutation and we can compute a distance
        float val = (*metric)( n , A , B , output );
        if ( val < opt_val ){
            opt_val = val;
            copyvec( bestperm , output , n );
        }
        
        v[ 0 ] = v[ 0 ] + 1;
        for (int  i = 0; i < n-1 ; i++){
            if( v[i] > n ){
                v[i] = i+1; 
                v[ i + 1 ] = v[ i + 1 ] + 1;
            }
        }
        
    }
    
    free(output);
    free(v);
    
    return opt_val;
}


// this function transforms and index into a permutation
// the function requires a bit of scrap space
__device__ __host__ void index_to_perm(lint r, int n, int *perm, int * scrap){
    
    for (int i = n ; i >=1; i--){
        scrap[n - i] =   (r % i) + 1;
        r = r/i;
    }

    // note that the way we are going the swap here is a bit different because
    // the elements from v are not in increasing form like in the cpu code.
    // In the parallel code the scrap is in the form
    // 1 1 1 , 2 1 1, 3 1 1, 1 2 1, 2 2 1 , 3 2 1
    // but in the serial code it is
    // 1 2 3 , 2 2 3  , 3 2 3, 1 3 3 , 2 3 3 , 3 3 3
    
    
    settoconsec(perm, n);
    for( int i = 0; i < n ; i++){
        swap(perm, i, i + scrap[i]-1);
    }
    
}

inline int fact(int n){
 if (n <=1)
     return 1;
 else
     return n*fact(n-1);
}

// this computes the optimal matching my testing different permutations in parallel
// we pass the nfact from outside to save time
// we cannot store the result of all evaluations in memory and then do a parallel max.
// there is just too much stuff to try. So each thread needs to keep a local max of several trials
__global__ void kernel_to_compute_optimal_match(int chunck_per_cycle, int num_perm_per_thread, lint nfact, int n, float *A, float *B, float (*metric)(int , float* , float *, int* ), float * obj_vals, lint * obj_perms ){
    
    int baseix = blockIdx.x*blockDim.x + threadIdx.x;;
    lint ix = baseix;
    
    // we copy A and B to shared memory because it might be faster when we are computing the norms
    extern __shared__ float AB_shared_mem[];
    // we need to split the shared memory into different parts
    float * shared_A = AB_shared_mem;
    float * shared_B = &AB_shared_mem[n*n];
    // the first thread of each block does the copy for the corresponding block
    if (threadIdx.x == 0){
        for (int i = 0; i < n*n ; i++){
            shared_A[i] = A[i];
            shared_B[i] = B[i];
        }
    }
    __syncthreads();
    
    float best_val = FLT_MAX;
    lint best_perm_ix;
    for (int i = 0; i < num_perm_per_thread ; i++){
        ix = baseix + chunck_per_cycle*i;
        // filter the stuff that does not matter
        if (ix < nfact){

            // probably we do not need more than 20 here
            int perm[MAX_N_PERM];
            int scrap[MAX_N_PERM];

            index_to_perm( ix ,  n, perm, scrap);

            float val = (*metric)( n,  shared_A ,  shared_B,  perm);
            if (val < best_val){
                best_val = val;
                best_perm_ix = ix;
            }

        }
    }
    
    obj_vals[baseix] = best_val;
    obj_perms[baseix] = best_perm_ix;
    
}


void test_index_perm(int n ){

    // test the function that indexes permutations sequentially
    
    int *perm  = (int *) malloc(n * sizeof(int));
    int *scrap = (int *) malloc(n * sizeof(int));
    
    for (int r = 0; r < fact(n) ; r++){
        index_to_perm(r, n, perm, scrap);
        //printvec(perm,n);
    }

    free(perm);
    free(scrap);
}



// this function will allocate space for A
float * read_graph_into_adj_mat(char * filename, int *graphsize, int directed){
    
    // if we are not given the graph size then we first read the file to try to estimate the size
    // of the graph by trying to find the largest index used
    // here we assume that the indices used are 1, 2, ..., n
    if (*graphsize == -1){
        FILE *  graphfile = fopen(filename , "r");
        
        
        int dim = -1;
        int edge1, edge2;
        while (   fscanf(graphfile, "%d %d\n", &edge1, &edge2) != EOF){
            if (dim < edge1){
                dim = edge1;
            }
            if (dim < edge2){
                dim = edge2;
            }
        }
        fclose(graphfile);
        *graphsize = dim;
        
    }
    
    // we use calloc because we want most of the entries to be zero and just have to set a few to non-zero
    // whatever edges are not specified in the file we are reading we will assume are zero
    float *A = (float *) calloc(  (*graphsize) , (*graphsize)*sizeof(float)    ) ;
    
    FILE *  graphfile = fopen(filename , "r");
    
    int edge1, edge2;
    while (   fscanf(graphfile, "%d %d\n", &edge1, &edge2) != EOF){
        if (edge1 <= (*graphsize) && edge2 <= (*graphsize) && edge1 >=1 && edge2 >=1 ){
            
            A[(edge1-1) + (edge2-1)* (*graphsize) ] = 1;
            
            // if the graph is undirected, we force it to be undirected
            if (directed == 0){
                    A[(edge2-1) + (edge1-1)* (*graphsize) ] = 1;
            }
        }
    }
    fclose(graphfile);

    return A;
    
}


// this writes a vector to an output file
void save_vec_to_file(int * vec, int n , char* output_file){
    
    // we only write if there is stuff to write. Otherwise we leave things as they are
    if (n > 1){
        FILE *  vec_file = fopen(output_file , "w");  

        for (int i = 0; i < n-1; i++){
            fprintf(vec_file,"%d ", vec[i]);
        }
        fprintf(vec_file,"%d", vec[n-1]);

        fclose(vec_file);
    }
}

int main(int argc,char *argv[]){
  
    
    if (argc != 8){
        printf("The arguments must be filenameA, filenameB, outputfile, L1vsL2, directed/undirected, gpu/cpu, size\n");
        return 0;
    }
    
    char * filenameA    = (char *) argv[1];
    char * filenameB    = (char *) argv[2];
    char * fileoutput   = (char *) argv[3];
    int norm_to_use     = atoi(    argv[4]  );
    int directed        = atoi(    argv[5]  );
    int cpu_vs_gpu      = atoi(    argv[6]  );
    int graphsize       = atoi(    argv[7]  );
    
    int sizeA = graphsize;
    int sizeB = graphsize;
    
    float *A = read_graph_into_adj_mat( filenameA , &sizeA , directed );
    float *B = read_graph_into_adj_mat( filenameB , &sizeB , directed );

    
    if ( sizeA != sizeB ){
        printf("Error, graphs of different sizes\n");
        return 0;
    }
    
    clock_t cpu_start, cpu_end;
    float cputime;
    
    int n = sizeA;
    
    lint nfact = fact(n);
    
    
    if (cpu_vs_gpu == 1){
        int * bestperm = (int *) malloc(n * sizeof( int )  ); //this is where we will keep the best perm
        cpu_start = clock();
        
        // we might want to try different norms
        float val;
        if (norm_to_use == 1){
        	val = compute_optimal_match(n, A, B, &L1_norm , bestperm);
        }
        if (norm_to_use == 2){
            val = compute_optimal_match(n, A, B, &fro_norm_square , bestperm);
        }
        
        cpu_end = clock();
        printf("CPU Opt Val = %f\n", val);
        printvec(bestperm, n);
        cputime = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
        printf("SERIAL: Time to compute opt =  %f\n",cputime); fflush(stdout);
        // store the vector in the output
        save_vec_to_file(bestperm,  n , fileoutput);
        // free the vector
        free(bestperm);
    }else{
    
        // now we have some GPU code
        cudaSetDevice( 0 );
        cudaDeviceReset();

        // here we compute the division of work
        // we try to make everything result in an iteger division of work
        int numthreadsperblock = 1024;
        int numblocks =  1024;
        int chunck_per_cycle = numblocks*numthreadsperblock;
        int num_stuff_per_thread = 1 + (nfact / chunck_per_cycle );
        //printf("Threads per block  = %d, Num blocks = %d , chunck_per_cycle = %d, num_stuff_per_thread = %d\n",numthreadsperblock,numblocks,chunck_per_cycle,num_stuff_per_thread);
        
        float * d_A;
        float * d_B;
        float * d_obj_vals;
        lint * d_obj_perms;

        float * h_obj_vals = (float *) malloc( chunck_per_cycle*sizeof(float) );
        lint * h_obj_perms = (lint *) malloc( chunck_per_cycle*sizeof(lint) );


        // create some timing variables
        cudaEvent_t gpu_start, gpu_end;
        float gputime;
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_end);

        cudaEventRecord(gpu_start, 0);
        cudaMalloc((void **)&d_A, n*n*sizeof(float) );
        cudaMalloc((void **)&d_B, n*n*sizeof(float) );
        cudaMalloc((void **)&d_obj_vals, chunck_per_cycle*sizeof(float) );
        cudaMalloc((void **)&d_obj_perms, chunck_per_cycle*sizeof(lint) );    
        cudaMemcpy( (void*) d_A , (void*) A , n*n*sizeof(float) , cudaMemcpyHostToDevice ); 
        cudaMemcpy( (void*) d_B , (void*) B , n*n*sizeof(float) , cudaMemcpyHostToDevice ); 
        cudaEventRecord(gpu_end, 0);
        cudaEventSynchronize(gpu_end);  //this is necessary to make sure that we measure time accurately. We can also use cudaDeviceSynchronize() but that is a bit of an overkill
        cudaEventElapsedTime(&gputime, gpu_start, gpu_end);
        printf ("PARALLEL: Time it took to allocate space: %f\n", gputime/1000); fflush(stdout);

        // this is the function pointer that we will pass to the GPU
        float (*h_d_per_metric)(int , float *, float * , int * );
        
        // we might want to use different norms
        if (norm_to_use == 1){
            cudaMemcpyFromSymbol(&h_d_per_metric, d_ptr_L1_norm, sizeof( float (*)(int , float *, float * , int * )  ));
        }
        if (norm_to_use == 2){
            cudaMemcpyFromSymbol(&h_d_per_metric, d_ptr_fro_norm_square, sizeof( float (*)(int , float *, float * , int * )  ));
        }
        
        cudaEventRecord(gpu_start, 0);
        kernel_to_compute_optimal_match<<<numblocks,numthreadsperblock,n*n*2*sizeof(float)>>>(chunck_per_cycle,num_stuff_per_thread , nfact,  n, d_A, d_B, h_d_per_metric , d_obj_vals, d_obj_perms);
        cudaEventRecord(gpu_end, 0);
        cudaEventSynchronize(gpu_end);  //this is necessary to make sure that we measure time accurately. We can also use cudaDeviceSynchronize() but that is a bit of an overkill
        cudaEventElapsedTime(&gputime, gpu_start, gpu_end);
        printf ("PARALLEL: Time it took to run the kernel: %f\n", gputime/1000); fflush(stdout);

        // now we copy the stuff back to the CPU and get the maximum by hand
        cudaEventRecord(gpu_start, 0);
        cudaMemcpy( (void*) h_obj_vals , (void*) d_obj_vals , chunck_per_cycle*sizeof(float) , cudaMemcpyDeviceToHost ); 
        cudaMemcpy( (void*) h_obj_perms , (void*) d_obj_perms , chunck_per_cycle*sizeof(float) , cudaMemcpyDeviceToHost ); 
        cudaEventRecord(gpu_end, 0);
        cudaEventSynchronize(gpu_end);  //this is necessary to make sure that we measure time accurately. We can also use cudaDeviceSynchronize() but that is a bit of an overkill
        cudaEventElapsedTime(&gputime, gpu_start, gpu_end);
        printf ("PARALLEL: Time it took to copy stuff back to the CPU: %f\n", gputime/1000); fflush(stdout);


        cpu_start = clock();
        float best_gpu_val = FLT_MAX;
        lint best_ix;
        for (int i = 0 ; i < chunck_per_cycle ; i++){
            float val = h_obj_vals[i];
            if (val < best_gpu_val){
                best_gpu_val  = val;
                best_ix = i;
            }
        }
        int * perm = (int *) malloc(n * sizeof(int));
        int * scrap = (int *) malloc(n * sizeof(int));
        index_to_perm(best_ix,  n, perm, scrap);
        printf("GPU Opt Val = %f\n", best_gpu_val);     
        printvec(perm, n);
        save_vec_to_file(perm,  n , fileoutput);
        cpu_end = clock();
        cputime = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
        printf("SERIAL: Time to compute the last step =  %f\n",cputime); fflush(stdout);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_obj_vals);
        cudaFree(d_obj_perms);
        
        free(h_obj_vals);
        free(h_obj_perms);
        free(perm);
        free(scrap);

        gerror( cudaPeekAtLastError() );
        cudaDeviceSynchronize();

    }
    
    
    free(A);
    free(B);
    
    
    return 0;
     
}
