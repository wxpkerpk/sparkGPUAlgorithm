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


__kernel void LogisticRegresion(
					__global  float* result,
					__global  float* data,
					__global  float* weight,
					__global  float* values,
					__global  int* Dim
					)
{
	int i = get_global_id(0);

	int dim = Dim[0];

	float sum = 0;
	for (int j = 0;j<dim; j++)
	{
		sum += data[i * dim + j] * weight[j];
	}
	sum = 1.0 / (1 + exp(-sum));
	for (int j = 0;j<dim; j++)
	{
		float a = (sum - values[i]) * data[i * dim + j] ;
		//result[j] += a;
		AtomicAdd(&result[j],a);
	}

}