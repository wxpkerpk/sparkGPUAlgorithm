package org.apache.spark.calabar.mllib.classification;

import org.apache.spark.calabar.Utils.OpenCLCodeUtil;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;
import com.calabar.chm.spark.utils.JOCLUtils.ClKernel;

import static org.jocl.CL.*;

/**
 * Created by wx on 2017/3/14.
 */
public class NaiveBayesClKernel {

    static  volatile   Long totalTime=0l;
    static String programSource;
    static String functionName="NaiveBayes";

    ClKernel clKernel;

    private   long maxItemsSize;


    private static NaiveBayesClKernel instance;
    public synchronized static NaiveBayesClKernel getInstance(){
        programSource= OpenCLCodeUtil.get(functionName);

        if(instance==null) instance=new NaiveBayesClKernel();

        return instance;

    }


    private  volatile boolean init= false;
    public NaiveBayesClKernel() {
        initialize();
    }
    public  void initialize() {
        programSource= OpenCLCodeUtil.get(functionName);

        clKernel=new ClKernel(functionName,programSource);

    }

    /**
     * Shut down and release all resources that have been allocated in
     * {@link #initialize()}
     */
    public  void shutdown() {
        clReleaseKernel(clKernel.kernel);
        clReleaseProgram(clKernel.program);
        clReleaseCommandQueue(clKernel.commandQueue);
        clReleaseContext(clKernel.context);
    }

    @Override
    protected void finalize() throws Throwable {
        shutdown();
        super.finalize();
    }

    public  int run(double []matrixArray,  double []piArray,double []piArray2,double []vectorArray,int dim,int k,int []outPutLabel)
    {


        if(piArray2==null) piArray2=new double[piArray.length];
        long start=System.currentTimeMillis();
        int []dims={dim};
        int []kCount={k};
        cl_mem inputMemB = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * matrixArray.length, Pointer.to(matrixArray), null);
        cl_mem inputMemC = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * piArray.length, Pointer.to(piArray), null);
        cl_mem inputMemC2= clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * piArray2.length, Pointer.to(piArray2), null);
        cl_mem inputMemD = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * vectorArray.length, Pointer.to(vectorArray), null);
        cl_mem inputK = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                , Pointer.to(kCount), null);
        cl_mem inputDim = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                , Pointer.to(dims), null);

        cl_mem outputCountMem = clCreateBuffer(clKernel.context,  CL_MEM_WRITE_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                * outPutLabel.length, Pointer.to(outPutLabel), null);
        int a = 0;
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(outputCountMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemB));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemC));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemC2));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemD));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputDim));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputK));
        clEnqueueNDRangeKernel(clKernel.commandQueue, clKernel.kernel, 1, null, new long[] { vectorArray.length/dim},
                null, 0, null, null);
         clFinish(clKernel.commandQueue);
        clEnqueueReadBuffer(clKernel.commandQueue, outputCountMem, CL_TRUE, 0,outPutLabel.length* Sizeof.cl_int,
                Pointer.to(outPutLabel), 0, null, null);

        clReleaseMemObject(inputMemB);
        clReleaseMemObject(inputMemC);
        clReleaseMemObject(inputMemD);
        clReleaseMemObject(outputCountMem);
        clReleaseMemObject(inputK);
        clReleaseMemObject(inputDim);
        long end=System.currentTimeMillis();
        synchronized (totalTime) {
            totalTime += end - start;
        }
        return 0;
    }

    public static void main(String []a)
    {




        long start1=System.currentTimeMillis();
        getInstance();
        long end1=System.currentTimeMillis();
        System.out.println(start1-end1);


    }




}
