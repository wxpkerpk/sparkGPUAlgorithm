package org.apache.spark.calabar.mllib.classification.logisticregresion;

import com.calabar.chm.spark.utils.JOCLUtils.ClKernel;
import com.calabar.chm.spark.utils.JOCLUtils.DeviceUtils;
import org.apache.spark.calabar.Utils.OpenCLCodeUtil;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import static org.jocl.CL.*;

/**
 * Created by wx on 2017/3/14.
 */
public class LogisticRegressionClKernel {

    static  volatile   Long totalTime=0l;
    static String programSource;
    static String functionName="LogisticRegresion";

    ClKernel clKernel;

    private   long maxItemsSize;


    cl_mem inputMemA ;
    cl_mem inputMemB ;
    cl_mem inputMemC ;
    int len=0;
    private  volatile boolean init= false;
    public LogisticRegressionClKernel(float []data,float [] values,int dim) {
        initialize();
        int []dims={dim};
        len=data.length/dim;
         inputMemA = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * data.length, Pointer.to(data), null);
         inputMemB = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * values.length, Pointer.to(values), null);
         inputMemC = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                , Pointer.to(dims), null);
        clSetKernelArg(clKernel.kernel, 1, Sizeof.cl_mem, Pointer.to(inputMemA));
        clSetKernelArg(clKernel.kernel,3, Sizeof.cl_mem, Pointer.to(inputMemB));
        clSetKernelArg(clKernel.kernel,4, Sizeof.cl_mem, Pointer.to(inputMemC));
    }
    public void clear()
    {

        clReleaseMemObject(inputMemB);
        clReleaseMemObject(inputMemC);
        clReleaseMemObject(inputMemA);
        shutdown();
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

    public  float[] run(float []weights)
    {



        float[] out=new float[weights.length];
        cl_mem inputMemD = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * weights.length, Pointer.to(weights), null);
        int workDims= DeviceUtils.getWorkDims(len,512);


        cl_mem outputCountMem = clCreateBuffer(clKernel.context,  CL_MEM_WRITE_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * out.length, Pointer.to(out), null);
        int a = 0;
        clSetKernelArg(clKernel.kernel, 0, Sizeof.cl_mem, Pointer.to(outputCountMem));
        clSetKernelArg(clKernel.kernel, 2, Sizeof.cl_mem, Pointer.to(inputMemD));
        clEnqueueNDRangeKernel(clKernel.commandQueue, clKernel.kernel, 1, null, new long[] { len},
                new long[]{workDims}, 0, null, null);
         clFinish(clKernel.commandQueue);
        clEnqueueReadBuffer(clKernel.commandQueue, outputCountMem, CL_TRUE, 0,out.length* Sizeof.cl_float,
                Pointer.to(out), 0, null, null);

        clReleaseMemObject(inputMemD);
        clReleaseMemObject(outputCountMem);
        long end=System.currentTimeMillis();

        return out;
    }

    public static void main(String []a)
    {







        long start1=System.currentTimeMillis();
        long end1=System.currentTimeMillis();
        System.out.println(start1-end1);


    }




}
