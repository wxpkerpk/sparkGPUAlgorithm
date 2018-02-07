package org.apache.spark.calabar.mllib.clustering.kmeans;

import com.calabar.chm.spark.utils.JOCLUtils.DeviceUtils;
import org.apache.spark.calabar.Utils.OpenCLCodeUtil;
import org.jocl.*;
import com.calabar.chm.spark.utils.JOCLUtils.ClKernel;

import static org.jocl.CL.*;

/**
 * Created by wx on 2017/3/14.
 */
public class KMeansClKernel {

    static  volatile   Long totalTime=0l;
    static String programSource;
    static String functionName="KMeans";

    ClKernel clKernel;

    private   long maxItemsSize;



    private volatile static KMeansClKernel instance;
    public synchronized static KMeansClKernel getInstance(){
        return instance;

    }
    int runLength=0;
    cl_mem outputCountMem ;
    cl_mem inputMemB ;
    cl_mem inputMemC ;
    cl_mem inputK ;
    cl_mem dimMem ;
    private int maxItem=32;
    public void clear()
    {
        clReleaseMemObject(outputCountMem);
        clReleaseMemObject(inputMemB);
        clReleaseMemObject(inputMemC);
        clReleaseMemObject(inputK);
        clReleaseMemObject(dimMem);




    }

    private  volatile boolean init= false;
    int []outPutCount;
     KMeansClKernel(float []inputPoints,float []inputNorm,int dim,int k)
     {
         initialize();

         outPutCount=new int[inputNorm.length];

          outputCountMem = clCreateBuffer(clKernel.context,  CL_MEM_WRITE_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                 * outPutCount.length, Pointer.to(outPutCount), null);
          inputMemB = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                 * inputPoints.length, Pointer.to(inputPoints), null);
          inputMemC = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                 * inputNorm.length, Pointer.to(inputNorm), null);
          inputK = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                 , Pointer.to(new int[]{k}), null);
          dimMem = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                 , Pointer.to(new int[]{dim}), null);
         runLength=inputNorm.length/dim;
         clSetKernelArg(clKernel.kernel, 1, Sizeof.cl_mem, Pointer.to(inputMemB));
         clSetKernelArg(clKernel.kernel, 3, Sizeof.cl_mem, Pointer.to(inputMemC));
         clSetKernelArg(clKernel.kernel, 5, Sizeof.cl_mem, Pointer.to(inputK));
         clSetKernelArg(clKernel.kernel, 6, Sizeof.cl_mem, Pointer.to(dimMem));
         clSetKernelArg(clKernel.kernel, 0, Sizeof.cl_mem, Pointer.to(outputCountMem));


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

    public  int[] run(float []clusters,  float []clustersNorm)
    {
//     KMeans(outPutCount,inputsNorm,clustersNorm,new int[]{k},new int[]{dim});


        long start=System.currentTimeMillis();





//        long currentParamsSize= DeviceUtils.caculateSize(kCount,dims,inputsNorm,clustersNorm,outPutCount);
//        long size=DeviceUtils.getMaxAllocSize(this.clKernel.device);
//       if(currentParamsSize>size) throw new IllegalArgumentException("超过显存最大限制"+size);



//       KMeans(outPutCount,inputsNorm,clustersNorm,kCount,dims);
        cl_mem inputMemB = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * clusters.length, Pointer.to(clusters), null);
        cl_mem inputMemC = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * clustersNorm.length, Pointer.to(clustersNorm), null);

        clSetKernelArg(clKernel.kernel, 2, Sizeof.cl_mem, Pointer.to(inputMemB));
        clSetKernelArg(clKernel.kernel, 4, Sizeof.cl_mem, Pointer.to(inputMemC));



        int workDims= DeviceUtils.getWorkDims(runLength,maxItem);

        clEnqueueNDRangeKernel(clKernel.commandQueue, clKernel.kernel, 1, null, new long[] {runLength},
                new long[] {workDims}, 0, null, null);
         clFinish(clKernel.commandQueue);
        clEnqueueReadBuffer(clKernel.commandQueue, outputCountMem, CL_TRUE, 0,outPutCount.length* Sizeof.cl_int,
                Pointer.to(outPutCount), 0, null, null);


        clReleaseMemObject(inputMemB);
        clReleaseMemObject(inputMemC);
        long end=System.currentTimeMillis();
        System.out.println(end-start);
        synchronized (totalTime) {
            totalTime += end - start;
        }
        return outPutCount;
    }


    void KMeans(
             int[] nOutCount,//每类的个数
             float[] vecInPointModel,//所有点的模
             float[]vecInCenterModel,//所有的中心点的模
             int[]InCenterNum,//中心点的个数
             int []dim
    )
    {
        int len=vecInPointModel.length/dim[0];
        for(int i=0;i<len;i++) {
            int dims=dim[0];
            int centerNum=InCenterNum[0];

            float minDistance = 9999999.9f;
            int flag = 0;
            for (int k = 0; k < centerNum; k++) {
                float dis=0;
                for(int j=0;j<dims;j++){
                    float l=vecInPointModel[i*dims+j]-vecInCenterModel[k*dims+j];
                    dis+=l*l;
                }
                if (dis <= minDistance) {
                    flag = k;
                    minDistance = dis;
                }
            }
            nOutCount[i] = flag;
        }
    }

    public static void main(String []a)
    {


        long start1=System.currentTimeMillis();
        int max=DeviceUtils.getWorkDims(1022,256);
        long end1=System.currentTimeMillis();
        System.out.println(start1-end1);


    }




}
