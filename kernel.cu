
#include <stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include<cublas_v2.h>
#include<cusparse.h>

#include<stdlib.h>
#include<time.h>



#define R_SIZE BLOCK_NUM * THREAD_NUM
#define M_SIZE R_SIZE * R_SIZE

#define FILE_BUFFER_LENGTH 30000









void get_rand_sparse(int m, int n, float** A, int percent)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = 0.0;

    int nnzNum = n * m * percent / 100;
    int* row = (int*)malloc(sizeof(int) * nnzNum);
    int* clo = (int*)malloc(sizeof(int) * nnzNum);
    float* value = (float*)malloc(sizeof(float) * nnzNum);


    //产生非零元

    srand((unsigned)time(NULL) + 23);
    for (int i = 0; i < nnzNum; i++)
    {
        row[i] = rand() % m;
    }

    srand((unsigned)time(NULL) + 43);
    for (int i = 0; i < nnzNum; i++)
    {
        clo[i] = rand() % n;
    }

    srand((unsigned)time(NULL) + 67);
    for (int i = 0; i < nnzNum; i++)
    {
        value[i] = (rand() % 10000) / 100.0 + 1.0;
    }

    for (int i = 0; i < nnzNum; i++)
    {
        A[row[i]][clo[i]] = value[i];



    }

    int nnz = 0;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            if (A[i][j] != 0)
                nnz++;
        }
    float per = nnz / (float)(m * n);
    printf("产生稀疏度为%f的矩阵\n", per);

}



void crsmv(float** A, float* A_first_row)
{
    float* clo_B1_dim1 = (float*)malloc(sizeof(float) * 784 * 300);
    for (int i = 0; i < 784; i++)
        for (int j = 0; j < 300; j++)
        {
            clo_B1_dim1[i * 300 + j] = A[j][i];


        }//300*784 列展开


    int* nnzperrow = (int*)malloc(sizeof(int) * 300);

    int* GPU_nnzperrow;

    int nnztotal;

    float* GPU_cusparse_B;
    cudaMalloc((void**)&GPU_cusparse_B, sizeof(float) * 300 * 784);
    cudaMalloc((void**)&GPU_nnzperrow, sizeof(int) * 300);

    cusparseHandle_t     handle = 0;//创建句柄
    cusparseMatDescr_t matb = 0;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&matb);
    cusparseSetMatType(matb, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matb, CUSPARSE_INDEX_BASE_ZERO);


    cudaMemcpy(GPU_cusparse_B, clo_B1_dim1, sizeof(float) * 300 * 784, cudaMemcpyHostToDevice);

    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, 300, 784, matb, GPU_cusparse_B, 300, GPU_nnzperrow, &nnztotal);//计算每一行非零元

    //printf("csr_nnnz:%d\n",nnztotal);
    cudaMemcpy(nnzperrow, GPU_nnzperrow, sizeof(int) * 300, cudaMemcpyDeviceToHost);

    //printf("NNZ 总数：%d ,", nnztotal);
    //printf("\n");
    //float epr = nnztotal / (300.00 * 784.00);
    //printf("NNZ 百分比：%f ,", epr);
    //printf("\n");

    float* GPU_CsrValB;
    int* GPU_csrRowPtrB;
    int* GPU_csrColIndB;

    cudaMalloc((void**)&GPU_CsrValB, sizeof(float) * nnztotal);
    cudaMalloc((void**)&GPU_csrColIndB, sizeof(int) * nnztotal);
    cudaMalloc((void**)&GPU_csrRowPtrB, sizeof(int) * (300 + 1));


    cusparseSdense2csr(handle,
        300,
        784,
        matb,
        GPU_cusparse_B,
        300,
        GPU_nnzperrow,
        GPU_CsrValB,
        GPU_csrRowPtrB,
        GPU_csrColIndB);



    float* GPU_cusparse_A_first_row;
    float* GPU_cusparse_resu;
    //float* cusparse_resu = (float*)malloc(sizeof(float) * 300);


    cudaMalloc((void**)&GPU_cusparse_A_first_row, sizeof(float) * 784);
    cudaMalloc((void**)&GPU_cusparse_resu, sizeof(float) * 300);

    cudaMemcpy(GPU_cusparse_A_first_row, A_first_row, sizeof(float) * 784, cudaMemcpyHostToDevice);

    float alpha = 1;
    float beta = 0;
    CUSPARSE_OPERATION_NON_TRANSPOSE;

    //double td1 = get_time();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    cusparseScsrmv(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        300,
        784,
        nnztotal,
        &alpha,
        matb,
        GPU_CsrValB,
        GPU_csrRowPtrB,
        GPU_csrColIndB,
        GPU_cusparse_A_first_row,
        &beta,
        GPU_cusparse_resu);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //cudaDeviceSynchronize();
   // double td2 = get_time();
    //td1 = td2 - td1;

    printf("CSR 运算时间：%f S", time);
    //printf("\n");

    //cudaMemcpy(cusparse_resu, GPU_cusparse_resu, sizeof(float) * 300, cudaMemcpyDeviceToHost);

    free(clo_B1_dim1);
    free(nnzperrow);
    cudaFree(GPU_cusparse_B);
    cudaFree(GPU_nnzperrow);
    cudaFree(GPU_CsrValB);
    cudaFree(GPU_csrColIndB);
    cudaFree(GPU_csrRowPtrB);
    cudaFree(GPU_cusparse_A_first_row);
    cudaFree(GPU_cusparse_resu);


}





void BSR_MV(float** B, float* ROW, int SIZE)


{
    /*---------------计算nnzb----------------------*/
    float** BSR_B = (float**)malloc(sizeof(float*) * 304);
    for (int i = 0; i < 304; i++)
    {
        BSR_B[i] = (float*)malloc(sizeof(float) * 784);

    }

    for (int i = 0; i < 304; i++)
        for (int j = 0; j < 784; j++)
            BSR_B[i][j] = 0;


    for (int i = 0; i < 300; i++)
        for (int j = 0; j < 784; j++)
            BSR_B[i][j] = B[i][j];


    int nnzb1 = 0;
    int Block_size = 4;
    int heart_row = 0;
    int heart_clo = 0;
    int flag = 0;
    for (int i = 0; i < 304 / Block_size; i++)
        for (int j = 0; j < 784 / Block_size; j++)
        {
            heart_row = i * Block_size;
            heart_clo = j * Block_size;

            flag = 0;//0表示块内没有非零元
            for (int i1 = 0; i1 < Block_size; i1++)
                for (int j1 = 0; j1 < Block_size; j1++)

                {
                    //printf("(%d, %d)\n",heart_row,heart_clo);
                    if (BSR_B[heart_row + i1][heart_clo + j1] != 0)
                        flag = 1;


                }
            if (flag == 1)
                nnzb1 = nnzb1 + 1;


        }


    printf(" nnzb1：%d\n", nnzb1);


    /*----------------------先计算csr格式------------------------------*/

    float* clo_B1_dim1 = (float*)malloc(sizeof(float) * 784 * 304);
    for (int i = 0; i < 784; i++)
        for (int j = 0; j < 304; j++)
        {
            clo_B1_dim1[i * 304 + j] = BSR_B[j][i];


        }//304*784 列展开


    int* nnzperrow = (int*)malloc(sizeof(int) * 304);

    int* GPU_nnzperrow;

    int nnztotal;

    float* GPU_cusparse_B;
    cudaMalloc((void**)&GPU_cusparse_B, sizeof(float) * 304 * 784);
    cudaMalloc((void**)&GPU_nnzperrow, sizeof(int) * 304);

    cusparseHandle_t     handle = 0;//创建句柄
    cusparseMatDescr_t matb = 0;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&matb);
    cusparseSetMatType(matb, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matb, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t matc = 0;
    cusparseCreateMatDescr(&matc);
    cusparseSetMatType(matc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matc, CUSPARSE_INDEX_BASE_ZERO);


    cudaMemcpy(GPU_cusparse_B, clo_B1_dim1, sizeof(float) * 304 * 784, cudaMemcpyHostToDevice);

    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, 304, 784, matb, GPU_cusparse_B, 304, GPU_nnzperrow, &nnztotal);//计算每一行非零元


    cudaMemcpy(nnzperrow, GPU_nnzperrow, sizeof(int) * 304, cudaMemcpyDeviceToHost);

    // printf("NNZ 总数：%d ,", nnztotal);
    //  printf("\n");
     // float epr = nnztotal / (300.00 * 784.00);
     // printf("NNZ 百分比：%f ,", epr);
     // printf("\n");

    float* GPU_CsrValB;
    int* GPU_csrRowPtrB;
    int* GPU_csrColIndB;

    cudaMalloc((void**)&GPU_CsrValB, sizeof(float) * nnztotal);
    cudaMalloc((void**)&GPU_csrColIndB, sizeof(int) * nnztotal);
    cudaMalloc((void**)&GPU_csrRowPtrB, sizeof(int) * (304 + 1));


    cusparseSdense2csr(handle,
        304,
        784,
        matb,
        GPU_cusparse_B,
        304,
        GPU_nnzperrow,
        GPU_CsrValB,
        GPU_csrRowPtrB,
        GPU_csrColIndB);


    /*----------------------------crs转bsr------------------------*/

    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
    int base, nnzb;
    int m = 304;
    int n = 784;
    int rowBlockDim = SIZE;
    int colBlockDim = SIZE;
    int mb = (m + rowBlockDim - 1) / rowBlockDim;
    int nb = (n + colBlockDim - 1) / colBlockDim;
    int bufferSize;
    void* pBuffer;
    int* bsrRowPtrC;

    cusparseScsr2gebsr_bufferSize(handle, dir, m, n,
        matb, GPU_CsrValB, GPU_csrRowPtrB, GPU_csrColIndB,
        rowBlockDim, colBlockDim,
        &bufferSize);

    cudaMalloc((void**)&pBuffer, bufferSize);
    cudaMalloc((void**)&bsrRowPtrC, sizeof(int) * (mb + 1));

    int* nnzTotalDevHostPtr = &nnzb;
    cusparseXcsr2gebsrNnz(handle, dir, m, n,
        matb, GPU_csrRowPtrB, GPU_csrColIndB,
        matc, bsrRowPtrC, rowBlockDim, colBlockDim,
        nnzTotalDevHostPtr,
        pBuffer);

    if (NULL != nnzTotalDevHostPtr) {
        nnzb = *nnzTotalDevHostPtr;
    }
    else {
        cudaMemcpy(&nnzb, bsrRowPtrC + mb, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base, bsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzb -= base;
    }
    // printf("nnz_per:%f\n", nnzb*SIZE*SIZE/(float)(784*304));

    printf("nnzb2  %d\n", nnzb);
    int* bsrColIndC;
    float* bsrValC;
    cudaMalloc((void**)&bsrColIndC, sizeof(int) * nnzb);
    cudaMalloc((void**)&bsrValC, sizeof(float) * (rowBlockDim * colBlockDim) * nnzb);
    cusparseScsr2gebsr(handle, dir, m, n,
        matb,
        GPU_CsrValB, GPU_csrRowPtrB, GPU_csrColIndB,
        matc,
        bsrValC, bsrRowPtrC, bsrColIndC,
        rowBlockDim, colBlockDim,
        pBuffer);


    /*---------------------BSR 相乘-----------------------*/
    float* x;
    float* GPU_BSR_resu;

    cudaMalloc((void**)&x, sizeof(float) * 784);
    cudaMalloc((void**)&GPU_BSR_resu, sizeof(float) * 304);
    cudaMemcpy(x, ROW, sizeof(float) * 784, cudaMemcpyHostToDevice);
    // cudaMemcpy(, hy, sizeof(float) * m, cudaMemcpyHostToDevice);

    float alpha = 1.0;
    float beta = 0;



    cusparseSbsrmv(handle, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &alpha,
        matc, bsrValC, bsrRowPtrC, bsrColIndC, SIZE, x, &beta, GPU_BSR_resu);
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cusparseSbsrmv(handle, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &alpha,
        matc, bsrValC, bsrRowPtrC, bsrColIndC, SIZE, x, &beta, GPU_BSR_resu);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("计算时间：%f\n", time);



    float* BSR_resu = (float*)malloc(sizeof(float) * 304);

    cudaMemcpy(BSR_resu, GPU_BSR_resu, sizeof(float) * 304, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 304; i++)
       //  printf("%f, ", BSR_resu[i]);
    // printf("\n");


      /*-------------释放空间--------------------*/


    cudaFree(GPU_cusparse_B);
    cudaFree(GPU_nnzperrow);
    cudaFree(GPU_CsrValB);
    cudaFree(GPU_csrColIndB);
    cudaFree(GPU_csrRowPtrB);
    cudaFree(pBuffer);
    cudaFree(bsrRowPtrC);
    cudaFree(pBuffer);
    cudaFree(bsrRowPtrC);

    free(BSR_resu);
    for (int i = 0; i < 304; i++)
        free(BSR_B[i]);
    free(BSR_B);
    free(clo_B1_dim1);

}

void BSR_MV_Large(float** B, float* ROW, int SIZE, int how_times)


{
    /*---------------计算nnzb----------------------*/
    float** BSR_B = (float**)malloc(sizeof(float*) * 304);
    for (int i = 0; i < 304; i++)
    {
        BSR_B[i] = (float*)malloc(sizeof(float) * 784 * how_times);

    }

    for (int i = 0; i < 304; i++)
        for (int j = 0; j < 784 * how_times; j++)
            BSR_B[i][j] = 0;

    for (int k = 0; k < how_times; k++)
        for (int i = 0; i < 300; i++)
            for (int j = 0; j < 784; j++)
                BSR_B[i][k * 784 + j] = B[i][j];


    int nnzb1 = 0;
    int Block_size = SIZE;
    int heart_row = 0;
    int heart_clo = 0;
    int flag = 0;
    /* for (int i = 0; i < 304 / Block_size; i++)
         for (int j = 0; j < 784 / Block_size; j++)
         {
             heart_row = i * Block_size;
             heart_clo = j * Block_size;

             flag = 0;//0表示块内没有非零元
             for (int i1 = 0; i1 < Block_size; i1++)
                 for (int j1 = 0; j1 < Block_size; j1++)

                 {
                     //printf("(%d, %d)\n",heart_row,heart_clo);
                     if (BSR_B[heart_row + i1][heart_clo + j1] != 0)
                         flag = 1;


                 }
             if (flag == 1)
                 nnzb1 = nnzb1 + 1;


         }


     printf(" nnzb1：%d\n", nnzb1);

     */
     /*----------------------先计算csr格式------------------------------*/

    float* clo_B1_dim1 = (float*)malloc(sizeof(float) * 784 * how_times * 304);
    for (int k = 0; k < how_times; k++)
        for (int i = 0; i < 784; i++)
            for (int j = 0; j < 304; j++)
            {
                clo_B1_dim1[k * 784 * 304 + i * 304 + j] = BSR_B[j][k * 784 + i];


            }//how_timeS*304*784 列展开


    int* nnzperrow = (int*)malloc(sizeof(int) * 304);

    int* GPU_nnzperrow;

    int nnztotal;

    float* GPU_cusparse_B;
    cudaMalloc((void**)&GPU_cusparse_B, sizeof(float) * 304 * 784 * how_times);
    cudaMalloc((void**)&GPU_nnzperrow, sizeof(int) * 304);

    cusparseHandle_t     handle = 0;//创建句柄
    cusparseMatDescr_t matb = 0;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&matb);
    cusparseSetMatType(matb, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matb, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t matc = 0;
    cusparseCreateMatDescr(&matc);
    cusparseSetMatType(matc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matc, CUSPARSE_INDEX_BASE_ZERO);


    cudaMemcpy(GPU_cusparse_B, clo_B1_dim1, sizeof(float) * 304 * 784 * how_times, cudaMemcpyHostToDevice);

    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, 304, 784 * how_times, matb, GPU_cusparse_B, 304, GPU_nnzperrow, &nnztotal);//计算每一行非零元


    cudaMemcpy(nnzperrow, GPU_nnzperrow, sizeof(int) * 304, cudaMemcpyDeviceToHost);

    // printf("NNZ 总数：%d ,", nnztotal);
    //  printf("\n");
     // float epr = nnztotal / (300.00 * 784.00);
     // printf("NNZ 百分比：%f ,", epr);
     // printf("\n");

    float* GPU_CsrValB;
    int* GPU_csrRowPtrB;
    int* GPU_csrColIndB;

    cudaMalloc((void**)&GPU_CsrValB, sizeof(float) * nnztotal);
    cudaMalloc((void**)&GPU_csrColIndB, sizeof(int) * nnztotal);
    cudaMalloc((void**)&GPU_csrRowPtrB, sizeof(int) * (304 + 1));


    cusparseSdense2csr(handle,
        304,
        784 * how_times,
        matb,
        GPU_cusparse_B,
        304,
        GPU_nnzperrow,
        GPU_CsrValB,
        GPU_csrRowPtrB,
        GPU_csrColIndB);


    /*----------------------------crs转bsr------------------------*/

    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
    int base, nnzb;
    int m = 304;
    int n = 784 * how_times;
    int rowBlockDim = SIZE;
    int colBlockDim = SIZE;
    int mb = (m + rowBlockDim - 1) / rowBlockDim;
    int nb = (n + colBlockDim - 1) / colBlockDim;
    int bufferSize;
    void* pBuffer;
    int* bsrRowPtrC;

    cusparseScsr2gebsr_bufferSize(handle, dir, m, n,
        matb, GPU_CsrValB, GPU_csrRowPtrB, GPU_csrColIndB,
        rowBlockDim, colBlockDim,
        &bufferSize);

    cudaMalloc((void**)&pBuffer, bufferSize);
    cudaMalloc((void**)&bsrRowPtrC, sizeof(int) * (mb + 1));

    int* nnzTotalDevHostPtr = &nnzb;
    cusparseXcsr2gebsrNnz(handle, dir, m, n,
        matb, GPU_csrRowPtrB, GPU_csrColIndB,
        matc, bsrRowPtrC, rowBlockDim, colBlockDim,
        nnzTotalDevHostPtr,
        pBuffer);

    if (NULL != nnzTotalDevHostPtr) {
        nnzb = *nnzTotalDevHostPtr;
    }
    else {
        cudaMemcpy(&nnzb, bsrRowPtrC + mb, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base, bsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzb -= base;
    }
    // printf("nnz_per:%f\n", nnzb * SIZE * SIZE / (float)(784*how_times * 304));
     printf("nnzb:%d\n", nnzb);
    int* bsrColIndC;
    float* bsrValC;
    cudaMalloc((void**)&bsrColIndC, sizeof(int) * nnzb);
    cudaMalloc((void**)&bsrValC, sizeof(float) * (rowBlockDim * colBlockDim) * nnzb);
    cusparseScsr2gebsr(handle, dir, m, n,
        matb,
        GPU_CsrValB, GPU_csrRowPtrB, GPU_csrColIndB,
        matc,
        bsrValC, bsrRowPtrC, bsrColIndC,
        rowBlockDim, colBlockDim,
        pBuffer);


    /*---------------------BSR 相乘-----------------------*/
    float* Large_Row = (float*)malloc(sizeof(float*) * 784 * how_times);
    for (int i = 0; i < how_times; i++)
        for (int j = 0; j < 784; j++)
            Large_Row[i * 784 + j] = ROW[j];


    float* x;
    float* GPU_BSR_resu;

    cudaMalloc((void**)&x, sizeof(float) * 784 * how_times);
    cudaMalloc((void**)&GPU_BSR_resu, sizeof(float) * 304);
    cudaMemcpy(x, Large_Row, sizeof(float) * 784 * how_times, cudaMemcpyHostToDevice);
    // cudaMemcpy(, hy, sizeof(float) * m, cudaMemcpyHostToDevice);

    float alpha = 1.0;
    float beta = 0;



    cusparseSbsrmv(handle, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &alpha,
        matc, bsrValC, bsrRowPtrC, bsrColIndC, SIZE, x, &beta, GPU_BSR_resu);
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cusparseSbsrmv(handle, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &alpha,
        matc, bsrValC, bsrRowPtrC, bsrColIndC, SIZE, x, &beta, GPU_BSR_resu);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("BSR块大小为%d计算时间：%f S\n", Block_size, time);



    float* BSR_resu = (float*)malloc(sizeof(float) * 304);

    cudaMemcpy(BSR_resu, GPU_BSR_resu, sizeof(float) * 304, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 304; i++)
        // printf("%f, ", BSR_resu[i]);
     //printf("\n");


      /*-------------释放空间--------------------*/


    cudaFree(GPU_cusparse_B);
    cudaFree(GPU_nnzperrow);
    cudaFree(GPU_CsrValB);
    cudaFree(GPU_csrColIndB);
    cudaFree(GPU_csrRowPtrB);
    cudaFree(pBuffer);
    cudaFree(bsrRowPtrC);
    cudaFree(pBuffer);
    cudaFree(bsrRowPtrC);

    free(BSR_resu);
    for (int i = 0; i < 304; i++)
        free(BSR_B[i]);
    free(BSR_B);
    free(clo_B1_dim1);

}

void get_block_matrix2(float** A, int m, int n, int Block_size, int percent)//假设A矩阵为304*784
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = 0;//置零

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            //printf("%f, ", A[i][j]);
        }
        // printf("\n");

    }

    //  printf("\n");


    int Row_size = m / Block_size;
    int Clo_size = n / Block_size;

    // printf("%d\n", Row_size);
    // printf("%d\n", Clo_size);


    int block_nnzNum = n * m * percent / (100 * Block_size * Block_size);
    // printf("块数%d\n", block_nnzNum);
    int* row = (int*)malloc(sizeof(int) * block_nnzNum);
    int* clo = (int*)malloc(sizeof(int) * block_nnzNum);
    float* value = (float*)malloc(sizeof(float) * block_nnzNum * Block_size * Block_size);


    //产生非零元

    srand((unsigned)time(NULL) + 23);
    for (int i = 0; i < block_nnzNum; i++)
    {
        row[i] = rand() % Row_size;
        // printf("%d, ", row[i]);
    }
    // printf("\n");

    srand((unsigned)time(NULL) + 43);
    for (int i = 0; i < block_nnzNum; i++)
    {
        clo[i] = rand() % Clo_size;
        //printf("%d, ", clo[i]);
    }
    //printf("\n");

    srand((unsigned)time(NULL) + 67);
    for (int i = 0; i < block_nnzNum * Block_size * Block_size; i++)
    {
        value[i] = (rand() % 10000) / 100.0 + 1.0;
        // printf("%f, ", value[i]);
    }
    // printf("\n");
    for (int i = 0; i < block_nnzNum; i++)
    {

        for (int j = 0; j < Block_size; j++)
            for (int k = 0; k < Block_size; k++)
                A[row[i] * Block_size + j][clo[i] * Block_size + k] = value[i * Block_size * Block_size + j * Block_size + k];





    }

    int nnz = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (A[i][j] != 0)
                nnz = nnz + 1;
        }


    }

    float per = nnz / (float)(m * n);
    printf("非零元所占百分比：%f\n", per);
    // printf("nnz：%d\n", nnz);
     //printf("nnzb1：%d\n", nnz/(Block_size*Block_size));
   // for (int i = 0; i < m; i++)
    //{
      //  for (int j = 0; j < n; j++)
        //{
          //  printf("%f,", A[i][j]);
        //}
        //printf("\n");

    //}

    //printf("\n");

    printf("生成块大小为%d的矩阵", Block_size);


}


void blas_mv(float** B, float* A_first_row)
{

    float* clo_B1_dim1 = (float*)malloc(sizeof(float) * 784 * 304);
    for (int i = 0; i < 784; i++)
        for (int j = 0; j < 304; j++)
        {
            clo_B1_dim1[i * 304 + j] = B[j][i];


        }//304*784 列展开




    int ida1 = 304;
    float alf = 1;
    float bet = 0;
    float* alpha1 = &alf;
    float* beta1 = &bet;
    float* cublas_resu = (float*)malloc(sizeof(float) * 304);
    float* GPU_cublas_resu;
    float* GPU_cusparse_B;
    float* GPU_cusparse_A_first_row;

    cudaMalloc((void**)&GPU_cublas_resu, sizeof(float) * 304);
    cudaMalloc((void**)&GPU_cusparse_B, sizeof(float) * 304 * 784);
    cudaMalloc((void**)&GPU_cusparse_A_first_row, sizeof(float) * 784);


    cudaMemcpy(GPU_cusparse_B, clo_B1_dim1, sizeof(float) * 304 * 784, cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_cusparse_A_first_row, A_first_row, sizeof(float) * 784, cudaMemcpyHostToDevice);

    cublasHandle_t handle1;
    cublasCreate(&handle1);

    cublasSgemv(handle1, CUBLAS_OP_N,
        304, 784,
        alpha1,
        GPU_cusparse_B, ida1,
        GPU_cusparse_A_first_row, 1,
        beta1,
        GPU_cublas_resu, 1);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    float time = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    cublasSgemv(handle1, CUBLAS_OP_N,
        304, 784,
        alpha1,
        GPU_cusparse_B, ida1,
        GPU_cusparse_A_first_row, 1,
        beta1,
        GPU_cublas_resu, 1);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    // cudaDeviceSynchronize();
     //double td2 = get_time();
     //td1 = td2 - td1;

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    printf("cublas 运算时间：%f", time);
    printf("\n");


    cudaMemcpy(cublas_resu, GPU_cublas_resu, sizeof(float) * 304, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 304; i++)
        printf("%f, ", cublas_resu[i]);
    printf("\n");



    free(clo_B1_dim1);


}
int main(int argc, char* argv[]) {



    float** rand_sparse;
    rand_sparse = (float**)malloc(sizeof(float*) * 300);
    for (int i = 0; i < 300; i++)
    {
        rand_sparse[i] = (float*)malloc(sizeof(float) * 784);

    }


   


    //读入文件picture

    float** A;
    A = (float**)malloc(sizeof(float*) * 50);
    for (int i = 0; i < 50; i++)
    {
        A[i] = (float*)malloc(sizeof(float) * 784);

    }

    FILE* fw = fopen("picture.txt", "r");

    for (int i = 0; i < 50; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            fscanf(fw, "%f", &A[i][j]);




        }




    }

    fclose(fw);
    //取A矩阵的第一行
    float* A_first_row = (float*)malloc(sizeof(float) * 784);
    for (int i = 0; i < 784; i++)
        A_first_row[i] = A[0][i];

    //读入文件layer1

    float** B1;
    B1 = (float**)malloc(sizeof(float*) * 300);
    for (int i = 0; i < 300; i++)
    {
        B1[i] = (float*)malloc(sizeof(float) * 784);

    }

    float** C;

    /*C = (float**)malloc(sizeof(float*) * 50);
    for (int i = 0; i < 50; i++)
    {
        C[i] = (float*)malloc(sizeof(float) * 300);

    }*/


    FILE* fw2 = fopen("layer1.txt", "r");

    for (int i = 0; i < 300; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            fscanf(fw, "%f", &B1[i][j]);




        }




    }

    fclose(fw2);

    //申请B矩阵，将B1转置

    float** B;
    B = (float**)malloc(sizeof(float*) * 784);
    for (int i = 0; i < 784; i++)
    {
        B[i] = (float*)malloc(sizeof(float) * 300);

    }
    for (int i = 0; i < 300; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            B[j][i] = B1[i][j];




        }




    }

    float** B_dense;
    B_dense = (float**)malloc(sizeof(float*) * 304);
    for (int i = 0; i < 304; i++)
    {
        B_dense[i] = (float*)malloc(sizeof(float) * 784);

    }
    float** B_sparse;
    B_sparse = (float**)malloc(sizeof(float*) * 304);
    for (int i = 0; i < 304; i++)
    {
        B_sparse[i] = (float*)malloc(sizeof(float) * 784);

    }
    float** B_ran;
    B_ran = (float**)malloc(sizeof(float*) * 304);
    for (int i = 0; i < 304; i++)
    {
        B_ran[i] = (float*)malloc(sizeof(float) * 784);

    }


    // get_block(B1, B_dense, B_sparse);



    // BSR_MV_Large(B_dense, A_first_row,16, 100);


     get_block_matrix2(B_ran,304,784, 4, 19);
     printf("\n");
     printf("----------------------------------\n");
     BSR_MV_Large(B_ran, A_first_row, 1, 20);
    // BSR_MV_Large(B_ran, A_first_row, 2, 10);
     BSR_MV_Large(B_ran, A_first_row, 4, 20);
    // BSR_MV_Large(B_ran, A_first_row, 8, 1);
    // crsmv(B_ran, A_first_row);
     printf("\n");
     printf("----------------------------------\n");
   //  printf("---------------blas-------------------\n");
    // blas_mv(B_ran, A_first_row);
    // printf("----------------------------------\n");

   
 
  


    for (int i = 0; i < 784; i++)
        free(B[i]);
    free(B);
    for (int i = 0; i < 300; i++)
        free(B1[i]);
    free(B1);
    for (int i = 0; i < 50; i++)
        free(A[i]);
    free(A);
    for (int i = 0; i < 300; i++)
        free(rand_sparse[i]);
    free(rand_sparse);
    for (int i = 0; i < 304; i++)
        free(B_dense[i]);
    free(B_dense);
    for (int i = 0; i < 304; i++)
        free(B_ran[i]);
    free(B_ran);
    for (int i = 0; i < 304; i++)
        free(B_sparse[i]);
    free(B_sparse);
   






}