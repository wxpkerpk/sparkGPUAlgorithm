package org.apache.spark.calabar.ml.recommendation;

import com.calabar.chm.spark.utils.JOCLUtils.ClKernel;
import com.calabar.chm.spark.utils.JOCLUtils.DeviceUtils;
import org.apache.spark.calabar.Utils.OpenCLCodeUtil;
import org.apache.spark.ml.recommendation.ALSWithGPU;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import static org.jocl.CL.*;
import com.github.fommil.netlib.BLAS;



/**
 * Created by wx on 2017/3/14.
 */
public class ALSClKernel {

    static  volatile   Long totalTime=0l;

    static String functionName="ALS";

    ClKernel clKernel;

    private   int  maxItemsSize=128;
   public   volatile long periods=0;


    private static ALSClKernel instance;
    public synchronized static ALSClKernel getInstance(){
        if(instance==null) instance=new ALSClKernel();
        return instance;

    }


    private  volatile boolean init= false;
    public ALSClKernel() {
        initialize();
    }
    public  void initialize() {
        String clCode=OpenCLCodeUtil.get(functionName);
        clKernel=new ClKernel(functionName,clCode);

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

    public   float[][] run(float []data,  float []ratins,int []perNum,float []lamdbaArrays,int rank,float [][]result)
    {



        long start=System.currentTimeMillis();

        long start1=System.currentTimeMillis();

        int trikLen=rank*(rank+1)/2;
        float []trik=new float[trikLen];
        float []atb=new float[rank];
        int []ranks={rank};



        float []output=new float[rank*lamdbaArrays.length];
        cl_mem dataMem = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * data.length, Pointer.to(data), null);
        cl_mem ratinsMem = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * ratins.length, Pointer.to(ratins), null);

        cl_mem perNumMem = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                * perNum.length, Pointer.to(perNum), null);
        cl_mem lamdbaArraysMem = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * lamdbaArrays.length, Pointer.to(lamdbaArrays), null);
        cl_mem ranksMem = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int
                * ranks.length, Pointer.to(ranks), null);
        cl_mem outputMem = clCreateBuffer(clKernel.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float
                * output.length, Pointer.to(output), null);
        long end1=System.currentTimeMillis();
        System.out.println("申请内存:"+(end1-start1));

        int a=0;
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(dataMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(ratinsMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(perNumMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(lamdbaArraysMem));
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(ranksMem));
        clSetKernelArg(clKernel.kernel, a++, trik.length*Sizeof.cl_float, null);
        clSetKernelArg(clKernel.kernel, a++,atb.length*Sizeof.cl_float, null);
        clSetKernelArg(clKernel.kernel, a++, trik.length*Sizeof.cl_float, null);
        clSetKernelArg(clKernel.kernel, a++, Sizeof.cl_mem, Pointer.to(outputMem));


        int workdim= DeviceUtils.getWorkDims(lamdbaArrays.length,maxItemsSize);
        long start2=System.currentTimeMillis();
        clEnqueueNDRangeKernel(clKernel.commandQueue, clKernel.kernel, 1, null, new long[] { lamdbaArrays.length},
                new long[]{1}, 0, null, null);

        clFinish(clKernel.commandQueue);
        clEnqueueReadBuffer(clKernel.commandQueue, outputMem, CL_TRUE, 0,output.length* Sizeof.cl_float,
                Pointer.to(output), 0, null, null);
        long end2=System.currentTimeMillis();
        System.out.println("运行:"+(end2-start2));
        for(int i=0;i<lamdbaArrays.length;i++){
            result[i]=new float[rank];
            System.arraycopy(output,i*rank,result[i],0,rank);
        }

        clReleaseMemObject(dataMem);
        clReleaseMemObject(ratinsMem);
        clReleaseMemObject(lamdbaArraysMem);
        clReleaseMemObject(ranksMem);
        clReleaseMemObject(outputMem);
        clReleaseMemObject(perNumMem);
        long end=System.currentTimeMillis();
        synchronized (this) {
            periods += end - start;
        }

        return  result;
    }


    //模拟运行分解矩阵
   static double []run(double []data,  double []ratins,int []perNum,double []lamdbaArrays,int rank,double []trik,double []atb,double []outputs)
    {

        for(int gid=0;gid<lamdbaArrays.length;gid++){
            int sum=0;
            for(int i=0;i<gid;i++){
                sum+=perNum[i];
            }
            for(int i=0;i<atb.length;i++){
                atb[i]=0;
            }
            for(int j=0;j<trik.length;j++){
                trik[j]=0;
            }
            for(int j=0;j<perNum[gid];j++){
                double []tempData=new double[rank];
                System.arraycopy(data,(sum+j)*rank,tempData,0,rank);
                BLAS.getInstance().dspr("U", rank,1.0, tempData, 1,trik );
                if(ratins[sum+j]!=0.0)  BLAS.getInstance().daxpy(rank, ratins[sum+j], tempData, 1, atb, 1);
            }
            ALSWithGPU.solve(trik,atb,lamdbaArrays[gid]);
            System.arraycopy(atb,0,outputs,(gid)*rank,rank);

        }

        return outputs;

    }

    public static void main(String []a)
    {


        int v1=5,v2=10;
        while(v1<v2-1){
            System.out.println(++v1);
        }
        String content=OpenCLCodeUtil.get("ALS");

        float []data={0.5655778f,
                0.8246948f,
                0.5655778f,
                0.8246948f};
        float []ratins={1f,1f};
        int []pernum={1,1};
        float [] lamdbas={0.01f,0.01f};
        int rank=2;
        float []trik=new float[3];
        float []atb=new float[2];
        float [][]output=new float[2][2];

        output= ALSClKernel.getInstance().run(data,ratins,pernum,lamdbas,rank,output);
        float []data2={ 0.5655778f,
                0.8246948f,
                0.5655778f,
                0.8246948f};
        float []ratins2={1f,1f};





    }




}
