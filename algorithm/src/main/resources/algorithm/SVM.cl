 void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source,
                             prevVal.intVal, newVal.intVal)
                             != prevVal.intVal);

		 }


__kernel void SVM(
					__global  float* gradient,
					__global  float* features,
					__global  float* weights,
					__global  float* values,
					__global  int* Dim
					)
{
	int i = get_global_id(0);

	int dim = Dim[0];

	float value = 2 * values[i] - 1.0;
    float dotProduct = 0.0;
    for (int j=0;j<dim;j++)
	{
		dotProduct += features[i * dim + j] * weights[j];
    }
    if (1.0 > value * dotProduct)
	{
		for (int k=0;k<dim;k++)
		{
			AtomicAdd(&gradient[k],-features[k + dim * i]*value);
        }
    }

}