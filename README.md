# exact_graph_match
This repository is for a GPU-based brute force exact small graph matching and distance computation.

It basically solves the problem 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;{\min_{P}}" title="\Large {\min_{P}}" />

, where $P$ is optimized over the set of all permutation matrices and $A$ and $B$ are the adjacency matrices of two graphs of equal size.



There are two functions, one is a C CUDA code and the other one is a Matlab wrapper to make it easier to run the code.

The C CUDA code can run on a regualar CPU with the right choice of flag but it will be slow. In any case, it should be compiled using nvcc.

The Matlab wrapper accepts as input two undirected graphs, the path to the compiled .cu file, and a flag, 0 or 1, depending on whether we want to a GPU or not. 

The CUDA code accepts as input two graph files, a flag indicating which norm to use, and a flag indicating wether we want to use a CPU or a GPU. In the CUDA code, the GPU device is by default set to 0. You can change this value in the code. The number of threads per block and the number of blocks are set to 1024 by default. You can change these values in the code.

Please cite this code using

Bento, J. and Ioannidis, S., 2018, May. A Family of Tractable Graph Distances. In Proceedings of the 2018 SIAM International Conference on Data Mining (pp. 333-341). Society for Industrial and Applied Mathematics.`

@inproceedings{bento2018family,
  title={A Family of Tractable Graph Distances},
  author={Bento, Jose and Ioannidis, Stratis},
  booktitle={Proceedings of the 2018 SIAM International Conference on Data Mining},
  pages={333--341},
  year={2018},
  organization={SIAM}
}
