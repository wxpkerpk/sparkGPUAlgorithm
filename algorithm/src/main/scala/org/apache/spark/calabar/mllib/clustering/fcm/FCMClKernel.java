package org.apache.spark.calabar.mllib.clustering.fcm;

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
public class FCMClKernel {

    static  volatile   Long totalTime=0l;
    static String programSource;
    static String functionName="FCM";

    ClKernel clKernel;

    private   long maxItemsSize;



    private volatile static FCMClKernel instance;
    public synchronized static FCMClKernel getInstance(){
        if(instance==null) instance=new FCMClKernel();
        return instance;

    }

    private int maxItem=256;

    private  volatile boolean init= false;
     FCMClKernel() {
        initialize();
    }
      void initialize() {
         programSource= OpenCLCodeUtil.get(functionName);

          clKernel=new ClKernel(functionName,programSource);
          maxItem= (int) DeviceUtils.getLong(clKernel.device, CL_DEVICE_MAX_WORK_GROUP_SIZE);


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

    public  float[] run(float []inputsNorm,  float []clustersNorm,int dim,int k)
    {
        long start=System.currentTimeMillis();
        int []kCount={k};
        int []dims={dim};
        float []outputU=new float[inputsNorm.length*k/dim];
        cl_mem inputMemB = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * inputsNorm.length, Pointer.to(inputsNorm), null);

        cl_mem inputMemC = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * clustersNorm.length, Pointer.to(clustersNorm), null);
        cl_mem inputK = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                , Pointer.to(kCount), null);
        cl_mem dimMem = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                , Pointer.to(dims), null);

        cl_mem outputCountMem = clCreateBuffer(clKernel.context,  CL_MEM_WRITE_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * outputU.length, Pointer.to(outputU), null);
        int a = 0;
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(outputCountMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemB));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputMemC));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(dimMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(inputK));


        int workDims= DeviceUtils.getWorkDims(clustersNorm.length/dim,maxItem);

        if(workDims==1) workDims=2;
        clEnqueueNDRangeKernel(clKernel.commandQueue, clKernel.kernel, 2, null, new long[] { inputsNorm.length/dim,k},
                new long[] {workDims,2}, 0, null, null);
         clFinish(clKernel.commandQueue);
        clEnqueueReadBuffer(clKernel.commandQueue, outputCountMem, CL_TRUE, 0,outputU.length* Sizeof.cl_float,
                Pointer.to(outputU), 0, null, null);


        clReleaseMemObject(inputMemB);
        clReleaseMemObject(inputMemC);
        clReleaseMemObject(outputCountMem);
        clReleaseMemObject(inputK);
        clReleaseMemObject(dimMem);
        long end=System.currentTimeMillis();
        System.out.println(end-start);
        return outputU;
    }


    public static void main(String []a)
    {



    }




}
