	#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void Canopy(
__global double* Outdis,
__global double* InCenter,
__global double* InPoint,
__global int* Dim
  )
{
    int gid = get_global_id(0);
	double sum = 0;
	int dims=Dim[0];
	for(int i=0;i<dims;i++)
	{
		//sum += (InPoint[i] - InCenter[i])*(InPoint[i] - InCenter[i]);
		//sum += fabs(InPoint[i] - InCenter[i]);
		double l = InPoint[i+gid*dims] - InCenter[i];
		sum += l*l;

	}
	Outdis[gid] = sqrt(sum);
}