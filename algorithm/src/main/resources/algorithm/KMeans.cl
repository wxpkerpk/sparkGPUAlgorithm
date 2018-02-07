__kernel void KMeans(
					 __global int* nOutCount,//每类的个数
					 __global float* vecInPointModel,//所有点
					 __global float* vecInCenterModel,//所有的中心点
					 __global float* vecInPointModel1,//所有点的模
					 __global float* vecInCenterModel1,//所有的中心点的模
					 __global int* InCenterNum,//中心点的个数
					 __global int* dim//

					  )
{
    int i = get_global_id(0);

	float minDistance = 9999999.9;
    int dims=dim[0];
    int centerNum=InCenterNum[0];
    int flag=0;

	for (int k = 0; k < centerNum; k++)
	{

		if( fabs(vecInPointModel1[i]-vecInCenterModel1[k]) < minDistance )
		{
				float dis=0;

				for(int j=0;j<dims;j++)
				{
					float l=vecInPointModel[i*dims+j]-vecInCenterModel[k*dims+j];
					dis+=l*l;
				}
				if (dis <= minDistance)
				{
					 flag = k;
					 minDistance = dis;
				}

		 }
	 }
     nOutCount[i] = flag;
}