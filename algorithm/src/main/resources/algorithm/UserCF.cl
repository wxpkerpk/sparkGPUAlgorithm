__kernel void UserCF(
					 __global double* nOutSemBlance,
					 __global double* vecInUserAll,
					   __global int* nCount
					  )
{
    int i = get_global_id(0);

	nOutSemBlance[i] = nCount[i]/(1+sqrt(vecInUserAll[i]));
}