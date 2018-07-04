# exact_graph_match
GPU based brute force exact small graph matching and distance computation


There are two functions, one is a C CUDA code and the other one is a Matlab wrapper to make it easier to run the code.

The C CUDA code can run on a regualar CPU with the right choice of flag but it will be slow. In any case, it should be compiled using nvcc.

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
