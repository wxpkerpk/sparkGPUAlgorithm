package org.apache.spark.calabar.mllib.clustering.Canopy;

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
public class CanopyClKernel {

    static  volatile   Long totalTime=0l;
    static String programSource ;
    static String functionName="Canopy";

    ClKernel clKernel;


    private   long maxItemsSize;


    private static CanopyClKernel instance;
    public synchronized static CanopyClKernel getInstance(){
        if(instance==null) instance=new CanopyClKernel();

        return instance;

    }


    private  volatile boolean init= false;
    public CanopyClKernel() {
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

    public  double [] run(final double []points, final double []clusters,final int dim)
    {


        int len=points.length/dim;
        long start=System.currentTimeMillis();
        double  []output=new double[len];
        int []dims={dim};
        cl_mem inputMemB = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * points.length, Pointer.to(points), null);
        cl_mem inputMemC = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * clusters.length, Pointer.to(clusters), null);
        cl_mem inputMemD = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                * dims.length, Pointer.to(dims), null);
        cl_mem outputCountMem = clCreateBuffer(clKernel.context,  CL_MEM_WRITE_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_double
                * output.length, Pointer.to(output), null);
        int a = 0;
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(outputCountMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemC));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemB));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemD));


        int workDim= DeviceUtils.getWorkDims(len,512);
        clEnqueueNDRangeKernel(clKernel.commandQueue, clKernel.kernel, 1, null, new long[] { len},
                new long[]{workDim}, 0, null, null);
         clFinish(clKernel.commandQueue);
        clEnqueueReadBuffer(clKernel.commandQueue, outputCountMem, CL_TRUE, 0,output.length* Sizeof.cl_double,
                Pointer.to(output), 0, null, null);
        clReleaseMemObject(inputMemB);
        clReleaseMemObject(inputMemC);
        clReleaseMemObject(inputMemD);
        clReleaseMemObject(outputCountMem);
        long end=System.currentTimeMillis();
        synchronized (totalTime) {
            totalTime += end - start;
        }
        return output;
    }

    public static void main(String []a)
    {




        long start1=System.currentTimeMillis();
        getInstance();
        long end1=System.currentTimeMillis();
        System.out.println(start1-end1);


    }




}
