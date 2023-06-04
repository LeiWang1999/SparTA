#ifndef _NMSPARSE_UTILS_H_
#define _NMSPARSE_UTILS_H_

#include "context.cuh"
#include <algorithm>

namespace nmsparse
{
    /**

    @brief Check if the given integer is equal to 1.
    This function takes an integer 'x' as input and checks if it is equal to 1. It returns a boolean value indicating the result of the comparison.
    @param x The integer to be checked.
    @return bool Returns true if 'x' is equal to 1, and false otherwise.
    */
    bool is_one(const int x)
    {
        return 1 == x;
    }

    /**
    @brief Check if the given integer is divisible by another integer.
    This function takes two integers, 'x' and 'be_devide', as input and checks if 'x' is divisible by 'be_devide'. It returns a boolean value indicating the result of the division (i.e., if the remainder is 0).
    @param x The integer to be divided.
    @param be_devide The integer by which 'x' is divided.
    @return bool Returns true if 'x' is divisible by 'be_devide', and false otherwise.
    */
    bool is_divisible(const int x, const int be_devide)
    {
        return 0 == (x % be_devide);
    }

    /**
    @brief Create a sparse representation in element-wise format from a dense matrix.
    This template function takes a dense matrix 'mat_in_dense' and converts it into a sparse representation in element-wise format. It populates the output arrays 'output_sparse_idx' and 'output_sparse_val' with the corresponding indices and values of the nonzero elements in 'mat_in_dense'. The function assumes that 'mat_in_dense' is of size k x n, where k represents the number of rows and n represents the number of columns.
    @param ctx The nmsparse context.
    @param k The number of rows in the dense matrix.
    @param n The number of columns in the dense matrix.
    @param mat_in_dense The input dense matrix.
    @param output_sparse_idx The output array for storing the row indices of nonzero elements.
    @param output_sparse_val The output array for storing the values of nonzero elements.
    @return bool Returns true if the conversion is successful, and false otherwise.
    */
    template <typename dtype>
    bool nmsparseCreateSparseElementWise(nmsparseContext_t ctx, int k, int n,
                                         dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val)
    {
        size_t idx = 0;
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                if (mat_in_dense[j * n + i] != 0)
                {
                    output_sparse_val[idx] = mat_in_dense[j * n + i];
                    output_sparse_idx[idx] = j;
                }
            }
        }
    }
    /**

    @brief Create a sparse representation in element-wise format from a dense matrix.
    This template function takes a dense matrix 'mat_in_dense' and converts it into a sparse representation in element-wise format. It populates the output arrays 'output_sparse_idx' and 'output_sparse_val' with the corresponding indices and values of the nonzero elements in 'mat_in_dense'. The function assumes that 'mat_in_dense' is of size k x n, where k represents the number of rows and n represents the number of columns.
    @param ctx The nmsparse context.
    @param k The number of rows in the dense matrix.
    @param n The number of columns in the dense matrix.
    @param mat_in_dense The input dense matrix.
    @param output_sparse_idx The output array for storing the row indices of nonzero elements.
    @param output_sparse_val The output array for storing the values of nonzero elements.
    @return bool Returns true if the conversion is successful, and false otherwise.
    */
    // TODO: implement the conversion from dense to vector-wise sparse pattern.
    template <typename dtype>
    bool nmsparseCreateSparseAlignN(nmsparseContext_t ctx, int k, int n,
                                    dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val, const int AlignN)
    {
        return false;
    }
    /**

    @brief Create a sparse representation from a dense matrix.
    This template function is a placeholder and currently returns true. It is intended to implement the conversion from a dense matrix to a sparse pattern, but the implementation is missing.
    @param ctx The nmsparse context.
    @param k The number of rows in the dense matrix.
    @param n The number of columns in the dense matrix.
    @param mat_in_dense The input dense matrix.
    @param output_sparse_idx The output array for storing the row indices of nonzero elements.
    @param output_sparse_val The output array for storing the values of nonzero elements.
    @return bool Always returns true.
    */
    template <typename dtype>
    bool nmsparseCreateSparse(nmsparseContext_t ctx, int k, int n,
                              dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val)
    {

        // check if the input is valid
        checkCtxPattern<dtype>(ctx);
        // TODO(lei): implement the conversion from dense to sparse pattern.

        return true;
    }
    /**

    @brief Initialize random data for element-wise sparse matrix representation.

    This template function generates random data for creating an element-wise sparse matrix representation. It populates the input vector 'vec' and the output arrays 'mat_data' and 'mat_index' with random values. The function supports generating data for multiple batches by setting the 'minibatch' parameter. The generated data follows a sparsity pattern defined by the 'sparsity' parameter.

    @param vec The input vector for random data generation.

    @param mat_data The output array for storing the values of the sparse matrix.

    @param mat_index The output array for storing the row indices of the sparse matrix.

    @param mat_data_for_gpu The output array for storing the values of the sparse matrix in GPU memory.

    @param mat_index_for_gpu The output array for storing the row indices of the sparse matrix in GPU memory.

    @param vecNum The size of the input vector.

    @param h The number of rows in the sparse matrix.

    @param sparsity The sparsity ratio of the generated matrix (between 0 and 1).

    @param minibatch The number of batches for generating random data.

    @tparam dtype The data type of the input vector and output matrix.
    */
    template <typename dtype>
    void nmSparseInitialRandomDataElementWise(dtype *vec, dtype *mat_data, int *mat_index, dtype *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparsity, int minibatch)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned)time(&t));
        unsigned int w = vecNum * (1.0f - sparsity);
        const int NUM_BANK = vecNum / 32;
        for (int batch = 0; batch < minibatch; ++batch)
            for (int i = 0; i < vecNum; ++i)
            {
                vec[i + vecNum * batch] = (dtype)rand() / RAND_MAX;
            }

        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
            {
                mat_data[i + j * w] = (dtype)rand() / RAND_MAX;
                mat_data_for_gpu[i * h + j] = mat_data[i + j * w];
            }

        int *tmp_index = (int *)malloc(vecNum / NUM_BANK * sizeof(int));
        for (int i = 0; i < vecNum / NUM_BANK; ++i)
            tmp_index[i] = i;

        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; i += w / NUM_BANK)
            {
                std::random_shuffle(tmp_index, tmp_index + vecNum / NUM_BANK);
                std::sort(tmp_index, tmp_index + w / NUM_BANK);
                for (int k = 0; k < w / NUM_BANK; ++k)
                {
                    mat_index[i + k + j * w] = tmp_index[k] + i / (1.0f - sparsity);
                    mat_index_for_gpu[(i + k) * h + j] = mat_index[i + k + j * w];
                }
            }
        }
        free(tmp_index);
    }

    /**

    @brief Initialize random data for vector-wise sparse matrix representation with alignment.

    This template function generates random data for creating a vector-wise sparse matrix representation with alignment. It populates the input vector 'vec' and the output arrays 'mat_data' and 'mat_index' with random values. The function supports generating data for multiple batches by setting the 'minibatch' parameter. The generated data follows a sparsity pattern defined by the 'sparsity' parameter. The alignment parameter 'ALIGN_N' determines the alignment size.

    @param vec The input vector for random data generation.

    @param mat_data The output array for storing the values of the sparse matrix.

    @param mat_index The output array for storing the row indices of the sparse matrix.

    @param mat_data_for_gpu The output array for storing the values of the sparse matrix in GPU memory.

    @param mat_index_for_gpu The output array for storing the row indices of the sparse matrix in GPU memory.

    @param vecNum The size of the input vector.

    @param h The number of rows in the sparse matrix.

    @param sparsity The sparsity ratio of the generated matrix (between 0 and 1).

    @param minibatch The number of batches for generating random data.

    @param ALIGN_N The alignment parameter.

    @tparam dtype The data type of the input vector and output matrix.
    */
    template <typename dtype>
    void nmSparseInitialRandomDataAlignN(dtype *vec, dtype *mat_data, int *mat_index, dtype *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparsity, int minibatch, const int ALIGN_N)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned)time(&t));
        unsigned int w = vecNum * (1.0f - sparsity);
        const int NUM_BANK = vecNum / 32;
        for (int batch = 0; batch < minibatch; ++batch)
            for (int i = 0; i < vecNum; ++i)
            {
                vec[i + vecNum * batch] = (dtype)rand() / RAND_MAX;
            }

        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
            {
                mat_data[i + j * w] = (dtype)rand() / RAND_MAX;
                mat_data_for_gpu[i * h + j] = mat_data[i + j * w];
            }

        int *tmp_index = (int *)malloc(vecNum / NUM_BANK * sizeof(int));
        for (int i = 0; i < vecNum / NUM_BANK; ++i)
            tmp_index[i] = i;

        for (int j = 0; j < h; j += ALIGN_N)
        {
            for (int i = 0; i < w; i += w / NUM_BANK)
            {
                std::random_shuffle(tmp_index, tmp_index + vecNum / NUM_BANK);
                std::sort(tmp_index, tmp_index + w / NUM_BANK);
                for (int k = 0; k < w / NUM_BANK; ++k)
                {
                    for (int j_in = 0; j_in < ALIGN_N; j_in += 1)
                    {
                        mat_index[i + k + (j + j_in) * w] = tmp_index[k] + i / (1.0f - sparsity);
                        mat_index_for_gpu[(i + k) * h + (j + j_in)] = mat_index[i + k + (j + j_in) * w];
                    }
                }
            }
        }
        free(tmp_index);
    }

    /**

    @brief Initialize random data for vector-wise sparse matrix representation with alignment.

    This template function generates random data for creating a vector-wise sparse matrix representation with alignment. It populates the input vector 'vec' and the output arrays 'mat_data' and 'mat_index' with random values. The function supports generating data for multiple batches by setting the 'minibatch' parameter. The generated data follows a sparsity pattern defined by the 'sparsity' parameter. The alignment parameters 'ALIGN_N' and 'ALIGN_K' determine the alignment sizes.

    @param vec The input vector for random data generation.

    @param mat_data The output array for storing the values of the sparse matrix.

    @param mat_index The output array for storing the row indices of the sparse matrix.

    @param mat_data_for_gpu The output array for storing the values of the sparse matrix in GPU memory.

    @param mat_index_for_gpu The output array for storing the row indices of the sparse matrix in GPU memory.

    @param vecNum The size of the input vector.

    @param h The number of rows in the sparse matrix.

    @param sparsity The sparsity ratio of the generated matrix (between 0 and 1).

    @param minibatch The number of batches for generating random data.

    @param ALIGN_N The alignment parameter for the row dimension.

    @param ALIGN_K The alignment parameter for the column dimension.

    @tparam dtype The data type of the input vector and output matrix.
    */
    template <typename dtype>
    void nmSparseInitialRandomDataAlignNAlignK(dtype *vec, dtype *mat_data, int *mat_index, dtype *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparsity, int minibatch, const int ALIGN_N, const int ALIGN_K)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned)time(&t));
        unsigned int w = vecNum * (1.0f - sparsity);
        const int NUM_BANK = vecNum / 32;
        for (int batch = 0; batch < minibatch; ++batch)
            for (int i = 0; i < vecNum; ++i)
            {
                vec[i + vecNum * batch] = (float)rand() / RAND_MAX;
                //	printf("%f\n", vec[i]);
            }

        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
            {
                mat_data[i + j * w] = (float)rand() / RAND_MAX;
                mat_data_for_gpu[i * h + j] = mat_data[i + j * w];
            }

        int *tmp_index = (int *)malloc(vecNum / NUM_BANK * sizeof(int));
        for (int i = 0; i < vecNum / NUM_BANK; ++i)
            tmp_index[i] = i;

        for (int j = 0; j < h; j += ALIGN_N)
        {
            for (int i = 0; i < w; i += w / NUM_BANK)
            {
                std::random_shuffle(tmp_index, tmp_index + vecNum / NUM_BANK);
                std::sort(tmp_index, tmp_index + w / NUM_BANK);
                for (int k = 0; k < w / NUM_BANK; k += ALIGN_K)
                {
                    for (int j_in = 0; j_in < ALIGN_N; j_in += 1)
                    {
                        for (int k_in = 0; k_in < ALIGN_K; k_in += 1)
                        {
                            mat_index[i + k + k_in + (j + j_in) * w] = tmp_index[k] + k_in + i / (1 - sparsity);
                            mat_index_for_gpu[(i + k + k_in) * h + (j + j_in)] = mat_index[i + k + k_in + (j + j_in) * w];
                        }
                    }
                }
            }
        }
        free(tmp_index);
    }

    /**
    @brief Initialize random data for element-wise sparse matrix representation.

    This template function generates random data for creating an element-wise sparse matrix representation. It populates the input vector 'vec' and the output arrays 'mat_data' and 'mat_index' with random values. The function supports generating data for multiple batches by setting the 'minibatch' parameter. The generated data follows a sparsity pattern defined by the 'sparsity' parameter.

    @param vec The input vector for random data generation.

    @tparam dtype The data type of the input vector and output matrix.
    */
    template <typename dtype>
    void nmSparseInitialRandomData(nmsparseContext_t ctx, dtype *vec, dtype *mat_data, int *mat_index, dtype *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparsity, int minibatch)
    {
        switch (ctx.nmsparsePattern)
        {
        case SparsePattern::ElementWise:
            nmSparseInitialRandomDataElementWise(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch);
            break;
        case SparsePattern::VectorWise4:
            nmSparseInitialRandomDataAlignN(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch, 4);
            break;
        case SparsePattern::VectorWise32:
            nmSparseInitialRandomDataAlignN(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch, 32);
            break;
        case SparsePattern::VectorWise64:
            nmSparseInitialRandomDataAlignN(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch, 64);
            break;
        case SparsePattern::BlockWise4x4:
            if (sparsity == 0.90625)
                nmSparseInitialRandomDataAlignNAlignK(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch, 4, 3);
            else
                nmSparseInitialRandomDataAlignNAlignK(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch, 4, 4);
            break;
        default:
            throw std::runtime_error("Unsupported sparse pattern");
            break;
        }
    }

    /**
    @brief Initialize random data for vector-wise sparse matrix representation with alignment.

    This template function generates random data for creating a vector-wise sparse matrix representation with alignment. It populates the input vector 'vec' and the output arrays 'mat_data' and 'mat_index' with random values. The function supports generating data for multiple batches by setting the 'minibatch' parameter. The generated data follows a sparsity pattern defined by the 'sparsity' parameter. The alignment parameter 'ALIGN_N' determines the alignment size.

    @param vec The input vector for random data generation.

    @tparam dtype The data type of the input vector and output matrix.
    */
    template <typename dtype>
    bool nmsparseCreateSparse(nmsparseContext_t ctx, int k, int n,
                              dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val);

    /**
    @brief Initialize random data for vector-wise sparse matrix representation with alignment.

    This template function generates random data for creating a vector-wise sparse matrix representation with alignment. It populates the input vector 'vec' and the output arrays 'mat_data' and 'mat_index' with random values. The function supports generating data for multiple batches by setting the 'minibatch' parameter. The generated data follows a sparsity pattern defined by the 'sparsity' parameter. The alignment parameters 'ALIGN_N' and 'ALIGN_K' determine the alignment sizes.

    @param vec The input vector for random data generation.

    @tparam dtype The data type of the input vector and output matrix.
    */
    template <typename dtype>
    void nmsparseCPURef_ALIGN_SHARED(dtype *vec, dtype *mat_data, int *mat_index, dtype *hostRef, const int condense_k, const int N, const int K, const int M)
    {
        float tmp;
        for (int batch = 0; batch < M; ++batch)
            for (int j = 0; j < N; ++j)
            {
                tmp = 0;
                for (int i = 0; i < condense_k; ++i)
                {
                    tmp += mat_data[i + j * condense_k] * vec[mat_index[i + j * condense_k] * M + batch];
                }
                hostRef[j * M + batch] = tmp;
            }
    }

    /**
    @brief Create a sparse matrix from a dense matrix.

    This template function creates a sparse matrix from a dense matrix. It takes the input dense matrix 'mat_in_dense' and outputs the sparse matrix indices 'output_sparse_idx' and values 'output_sparse_val' based on the specified data type 'dtype'.

    @param mat_in_dense The input dense matrix.
    @param output_sparse_idx The output array for sparse matrix indices.
    @param output_sparse_val The output array for sparse matrix values.

    @tparam dtype The data type of the input and output matrices.

    @return Returns true if the operation was successful, false otherwise.
    */
    template <typename dtype>
    void nmsparseCPURef(dtype *vec, dtype *mat_data, int *mat_index, dtype *hostRef, const int condense_k, const int N, const int K, const int M)
    {
        float tmp;
        for (int batch = 0; batch < M; ++batch)
            for (int j = 0; j < N; ++j)
            {
                tmp = 0;
                for (int i = 0; i < condense_k; ++i)
                {
                    tmp += mat_data[i + j * condense_k] * vec[mat_index[i + j * condense_k] + batch * K];
                }
                hostRef[j + batch * N] = tmp;
            }
    }

    /**
    @brief Check the result of a sparse matrix operation against a CPU reference implementation.

    This template function compares the results of a CPU reference implementation with the GPU implementation. It checks for element-wise matching within a specified epsilon tolerance and returns a boolean value indicating whether the results match.

    @param hostRef The CPU reference implementation result.
    @param gpuRef The GPU implementation result.
    @param M The number of batches.
    @param N The dimension of the matrix.

    @tparam dtype The data type of the input and output matrices.

    @return Returns true if the results match within the epsilon tolerance, false otherwise.
    */
    template <typename dtype>
    bool nmsparseCheckResult(dtype *hostRef, dtype *gpuRef, const int M, const int N)
    {
        double epsilon = 1E-2;
        bool match = 1;
        for (int batch = 0; batch < M; ++batch)
            for (int i = 0; i < N; i++)
            {
                if (abs((hostRef[i + batch * N] - gpuRef[i + batch * N]) / hostRef[i + batch * N]) > epsilon)
                {
                    match = 0;
                    printf("Arrays do [NOT] match!\n");
                    printf("host %5.5f gpu %5.5f at current %d\n", hostRef[i + batch * N], gpuRef[i + batch * N], i + batch * N);
                    break;
                }
            }
        return match;
    }
}

#endif
