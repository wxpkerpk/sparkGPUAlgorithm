
void dspr(int var1, float var2, __global float* var4, int var5, int var6, __local float* var7, int var8) ;
void daxpy(int var0, float var1, __global float* var3, int var4, int var5, __local float* var6, int var7, int var8);
void Cholesky(__local float* A,__local float* L,int n)  ;
void Solve(__local float* L,__local  float* X, int n)  ;



__kernel void ALS(
	 __global float* data,
	__global float* ratins,
	__global int* perNum,
	__global float* lamdbaArrays,
	__global int* rank,
	__local float* trik,
	__local float* atb,
	__local float* L,
	 __global float* OutPut
	)
{
	int gid = get_global_id(0);
    	for (int i_i=0;i_i<rank[0];i_i++)
    	{
    		atb[i_i] = 0;
    	}
    	for (int i_i=0;i_i<rank[0]*(rank[0]+1)/2;i_i++)
    	{
    		trik[i_i] = 0;
    		L[i_i] = 0;
    	}
    

    	int sum = 0;

    	for(int i=0;i<gid;i++)
    	{
    		sum += perNum[i];
    	}

    		for(int j=0;j<perNum[gid];j++)
    		{
    			dspr(rank[0],1.0, data,(sum+j)*rank[0],1,trik ,0);
    			if (ratins[sum+j] != 0.0)
    			{
    				daxpy(rank[0], ratins[sum+j], data, (sum+j)*rank[0],1, atb,0, 1);
    			}
    		}
    		int i_lamda = 0;
    		int j_lamda = 2;
    		while( i_lamda < rank[0]*(rank[0]+1)/2 )
    		{
    			trik[i_lamda] += lamdbaArrays[gid];
    			i_lamda += j_lamda;
    			j_lamda += 1;
    		}

    		int n = rank[0];//长度为10的数组 得出为4
    		Cholesky(trik, L, n);
    		Solve(L, atb ,n); //大小为4

    		for (int i_i=0;i_i<rank[0];i_i++)
    		{
    			OutPut[gid*rank[0]+i_i] = atb[i_i];
    			atb[i_i] = 0;
    		}
    		for (int i_i=0;i_i<rank[0]*(rank[0]+1)/2;i_i++)
    		{
    			L[i_i] = 0;
    		}
}

void Cholesky(__local float* A,__local float* L,int n)
{
    for(int k = 0; k < n; k++)
    {
        float sum = 0;
		int nkdex = k*(k+1)/2;
        for(int i = 0; i < k; i++)
		{
            sum += L[nkdex+i] * L[nkdex+i];
		}
		sum = A[nkdex+k] - sum;
        L[nkdex+k] = sqrt(sum > 0 ? sum : 0);
        for(int i = k + 1; i < n; i++)
        {
			int nidex = i*(i+1)/2;
            sum = 0;
            for(int j = 0; j < k; j++)
			{
                sum += L[nidex+j] * L[nkdex+j];
			}
            L[nidex+k] = (A[nidex+k] - sum) / L[nkdex+k];
        }
       // for(int j = 0; j < k; j++)
          // L[j][k] = 0;
    }
}


void Solve(__local float* L,__local float* X, int n)
{
    for(int k = 0; k < n; k++)
    {
		int nkdex = k*(k+1)/2;
        for(int i = 0; i < k; i++)
		{
            X[k] -= X[i] * L[nkdex+i];
		}
        X[k] /= L[nkdex+k];
    }

    for(int k = n - 1; k >= 0; k--)
    {
		int nkdex = k*(k+1)/2;
        for(int i = k + 1; i < n; i++)
		{
			int nidex = i*(i+1)/2;
            X[k] -= X[i] * L[nidex+k];
		}
        X[k] /= L[nkdex+k];
    }
}

void dspr(int var1, float var2, __global float* var4, int var5, int var6, __local float* var7, int var8)
{
	float var11 = 0.0;
	bool var13 = false;
	bool var14 = false;
	bool var15 = false;
	bool var16 = false;
	bool var17 = false;
	bool var18 = false;
	bool var19 = false;
	int var20 = 0;

	if(var1 != 0 && var2 != 0.0)
	{
		if(var6 <= 0)
		{
			var20 = 1 - (var1 - 1) * var6;
		}
		else if(var6 != 1)
		{
			var20 = 1;
		}

		int var29 = 1;
		int var21;
		int var22;
		int var23;
		int var25;
		int var26;
		int var27;
		int var28;

		if(var6 == 1)
		{
			var26 = 1;

			for(var21 = var1 - 1 + 1; var21 > 0; --var21)
			{
				if(var4[var26 - 1 + var5] != 0.0)
				{
					var11 = var2 * var4[var26 - 1 + var5];
					var28 = var29;
					var23 = 1;

					for(var22 = var26 - 1 + 1; var22 > 0; --var22)
					{
						var7[var28 - 1 + var8] += var4[var23 - 1 + var5] * var11;
						++var28;
						++var23;
					}
				}

				var29 += var26;
				++var26;
			}
		}
		else
		{
			var27 = var20;
			var26 = 1;

			for(var21 = var1 - 1 + 1; var21 > 0; --var21)
			{
				if(var4[var27 - 1 + var5] != 0.0)
				{
					var11 = var2 * var4[var27 - 1 + var5];
					var25 = var20;
					var28 = var29;

					for(var22 = var29 + var26 - 1 - var29 + 1; var22 > 0; --var22)
					{
						var7[var28 - 1 + var8] += var4[var25 - 1 + var5] * var11;
						var25 += var6;
						++var28;
					}
				}

				var27 += var6;
				var29 += var26;
				++var26;
			}
		}
	}
}
void daxpy(int var0, float var1, __global float* var3, int var4, int var5, __local float* var6, int var7, int var8)
{


	bool var9 = false;
	bool var10 = false;
	bool var11 = false;
	bool var12 = false;
	bool var13 = false;
	if(var0 > 0) {
		if(var1 != 0.0) {
			int var14;
			int var15;
			if(var5 == 1 && var8 == 1) {
				int var18 = var0 % 4;
				if(var18 != 0) {
					var15 = 1;

					for(var14 = var18 - 1 + 1; var14 > 0; --var14) {
						var6[var15 - 1 + var7] += var1 * var3[var15 - 1 + var4];
						++var15;
					}

					if(var0 < 4) {
						return;
					}
				}

				int var19 = var18 + 1;
				var15 = var19;

				for(var14 = (var0 - var19 + 4) / 4; var14 > 0; --var14) {
					var6[var15 - 1 + var7] += var1 * var3[var15 - 1 + var4];
					var6[var15 + 1 - 1 + var7] += var1 * var3[var15 + 1 - 1 + var4];
					var6[var15 + 2 - 1 + var7] += var1 * var3[var15 + 2 - 1 + var4];
					var6[var15 + 3 - 1 + var7] += var1 * var3[var15 + 3 - 1 + var4];
					var15 += 4;
				}

			} else {
				int var16 = 1;
				int var17 = 1;
				if(var5 < 0) {
					var16 = (-var0 + 1) * var5 + 1;
				}

				if(var8 < 0) {
					var17 = (-var0 + 1) * var8 + 1;
				}

				var15 = 1;

				for(var14 = var0 - 1 + 1; var14 > 0; --var14) {
					var6[var17 - 1 + var7] += var1 * var3[var16 - 1 + var4];
					var16 += var5;
					var17 += var8;
					++var15;
				}

			}
		}
	}

}