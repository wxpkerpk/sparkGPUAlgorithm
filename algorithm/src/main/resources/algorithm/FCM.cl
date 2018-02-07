__kernel void FCM(
					__global  float* Out,__global  float* In_X,__global  float* In_C,__global  int* Dim,__global  int* CenterNum
					)
{
	int X_n = get_global_id(0);
	int C_n = get_global_id(1);

	int DimNum = Dim[0];
	int centerN=CenterNum[0];
	float powNum = 2.0/(1.0*(2-1));//m=2
	float sumAll = 0;

	float sum1=0;
	for (int i=0;i<DimNum;i++)
	{
		sum1 += pow(In_X[X_n*DimNum+i]-In_C[C_n*DimNum+i],2);
	}


    float  alpha=0.00000001;
	float DisXC = sqrt(sum1)+alpha;
	for (int k=0;k<centerN;k++)
	{
		float sum2=0;
		for (int i=0;i<DimNum;i++)
		{
			sum2 += pow(In_X[X_n*DimNum+i]-In_C[k*DimNum+i],2);
		}
		float sum_b = sqrt(sum2)+alpha;
		sumAll += (DisXC/sum_b)*(DisXC/sum_b);
	}

	Out[CenterNum[0]*X_n+C_n] = 1/(sumAll*sumAll);

}



