
	#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void NaiveBayes(
					 __global int* nOutSumCount,
					 __global double* vecInPoint,
					 __global double* vecInPointCN,
					 __global double* vecInPointCN2,

					 __global double* vecIndata,
					   __global int* nDim,
					  __global int* nCN

					  )
{
    int i = get_global_id(0);

	double dbSum = 0;
	double dbMax = -9999;
	int flag = 0;
	for(int k=0;k<nCN[0];k++)
	{
		dbSum = 0;
		for(int j = 0;j<nDim[0];j++)
		{
			dbSum += vecIndata[i*nDim[0]+j]*vecInPoint[k*nDim[0]+j];
		}
		dbSum += vecInPointCN[k];
		dbSum += vecInPointCN2[k];


		if(dbMax < dbSum)
		{
			dbMax = dbSum;
			flag = k;
		}
	}
	nOutSumCount[i] = flag;

}