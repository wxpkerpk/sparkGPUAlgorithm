package org.apache.spark.calabar.mllib.recomment;

import com.calabar.chm.spark.utils.JOCLUtils.ClKernel;
import org.apache.spark.calabar.Utils.OpenCLCodeUtil;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import static org.jocl.CL.*;
/**
 * Created by wx on 2017/3/14.
 */
public class UserCfClKernel {

    static  volatile   Long totalTime=0l;
    static String programSource ;
    static String functionName="UserCF";

    ClKernel clKernel;


    private   long maxItemsSize;


    private static UserCfClKernel instance;
    public synchronized static UserCfClKernel getInstance(){
        if(instance==null) instance=new UserCfClKernel();

        return instance;

    }


    private  volatile boolean init= false;
    public UserCfClKernel() {
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

    public  double [] run(double []input1,  int []input2)
    {

        double []similarity=new double[input1.length];

        long start=System.currentTimeMillis();
        cl_mem inputMemB = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * input1.length, Pointer.to(input1), null);
        cl_mem inputMemC = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                * input2.length, Pointer.to(input2), null);


        cl_mem outputCountMem = clCreateBuffer(clKernel.context,  CL_MEM_WRITE_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * similarity.length, Pointer.to(similarity), null);
        int a = 0;
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(outputCountMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemB));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemC));

        clEnqueueNDRangeKernel(clKernel.commandQueue, clKernel.kernel, 1, null, new long[] { input1.length},
                null, 0, null, null);
         clFinish(clKernel.commandQueue);
        clEnqueueReadBuffer(clKernel.commandQueue, outputCountMem, CL_TRUE, 0,similarity.length* Sizeof.cl_double,
                Pointer.to(similarity), 0, null, null);

        clReleaseMemObject(inputMemB);
        clReleaseMemObject(inputMemC);
        clReleaseMemObject(outputCountMem);
        long end=System.currentTimeMillis();
        synchronized (totalTime) {
            totalTime += end - start;
        }
        return similarity;
    }

    public static void main(String []a)
    {




        long start1=System.currentTimeMillis();
        getInstance();
        long end1=System.currentTimeMillis();
        System.out.println(start1-end1);


    }




}
